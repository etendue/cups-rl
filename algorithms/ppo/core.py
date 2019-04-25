import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical


def count_vars(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def calculate_output_size_after_4_conv_layers(frame_dim, stride=2, kernel_size=3, padding=1,
                                              num_filters=32):
    """
    Assumes square resolution image. Find LSTM size after 4 conv layers below in A3C using regular
    Convolution math. For example:
    42x42 -> (42 − 3 + 2)÷ 2 + 1 = 21x21 after 1 layer
    11x11 after 2 layers -> 6x6 after 3 -> and finally 3x3 after 4 layers
    Therefore lstm input size after flattening would be (3 * 3 * num_filters)
    """

    width = (frame_dim - kernel_size + 2 * padding) // stride + 1
    width = (width - kernel_size + 2 * padding) // stride + 1
    width = (width - kernel_size + 2 * padding) // stride + 1
    width = (width - kernel_size + 2 * padding) // stride + 1

    return width * width * num_filters


def normalized_columns_initializer(weights, std=1.0):
    """
    Weights are normalized over their column. Also, allows control over std which is useful for
    initialising action logit output so that all actions have similar likelihood
    """

    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1, keepdim=True))
    return out


class CNNRNNBase(nn.Module):
    """
    Mainly Ikostrikov's implementation of A3C (https://arxiv.org/abs/1602.01783).

    Processes an input image (with num_input_channels) with 4 conv layers,
    interspersed with 4 elu activation functions. The output of the final layer is then flattened
    and passed to an LSTM (with previous or initial hidden and cell states (hx and cx)).
    The new hidden state is used as an input to the critic and value nn.Linear layer heads,
    The final output is then predicted value, action logits, hx and cx.
    """

    def __init__(self, input_shape, output_size):
        #  TODO: initialization weights and bias
        super(CNNRNNBase, self).__init__()
        ch, w, h = input_shape
        self.input_shape = input_shape
        self.conv1 = nn.Conv2d(ch, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)

        # assumes square image
        self.rnn_insize = calculate_output_size_after_4_conv_layers(w)
        self.rnn_size = output_size
        self.rnn = nn.GRUCell(self.rnn_insize, self.rnn_size)

    def forward(self, x, x_mask, h, horizon_t=1):

        # x has shape [batch,ch, w, h]
        # x_mask has shape [batch]
        # h has shape   [batch, hidden_state_size]

        if len(x.size()) == 3:  # if batch forgotten, with 1 time step
            x = x.unsqueeze(0)
            x_mask = x_mask.unsqueeze(0)
            h = h.unsqueeze(0)

        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))

        # convert back to time sequence for RNN cell
        x = x.view(horizon_t, -1, self.rnn_insize)
        x_mask = x_mask.view(horizon_t,-1,1)
        h = h.view(horizon_t, -1, self.rnn_size)
        outputs = []

        h = h[0]  # use only the start state
        for t in range(horizon_t):
            out = h = self.rnn(x[t], h * x_mask[t])
            outputs.append(out)
        outputs = torch.stack(outputs, dim=0)
        return outputs.view(-1, self.rnn_size), h.view(-1, self.rnn_size)


class MLP(nn.Module):
    def __init__(self,
                 layers,
                 activation=torch.tanh,
                 output_activation=None,
                 output_squeeze=False):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.activation = activation
        self.output_activation = output_activation
        self.output_squeeze = output_squeeze

        for i, layer in enumerate(layers[1:]):
            self.layers.append(nn.Linear(layers[i], layer))
            nn.init.zeros_(self.layers[i].bias)

    def forward(self, input):
        x = input
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        if self.output_activation is None:
            x = self.layers[-1](x)
        else:
            x = self.output_activation(self.layers[-1](x))
        return x.squeeze() if self.output_squeeze else x


class CategoricalPolicy(nn.Module):
    def __init__(self, in_features, hidden_sizes, activation,
                 output_activation, action_dim):
        super(CategoricalPolicy, self).__init__()

        self.logits = MLP(
            layers=[in_features] + list(hidden_sizes) + [action_dim],
            activation=activation)

    def forward(self, x, a=None):
        policy = Categorical(logits=self.logits(x))
        if a is None:
            a = policy.sample().squeeze()
        logp_a = policy.log_prob(a).squeeze()
        return a, logp_a, policy.entropy()


class ActorCritic(nn.Module):
    def __init__(self,
                 in_features,
                 action_space,
                 hidden_sizes=(64, 64),
                 memory_size = 256,
                 activation=torch.tanh,
                 output_activation=None):
        super(ActorCritic, self).__init__()

        self.feature_base = CNNRNNBase(
            input_shape=in_features,
            output_size=memory_size
        )

        self.policy = CategoricalPolicy(
                memory_size,
                hidden_sizes,
                activation,
                output_activation,
                action_dim=action_space.n)

        self.value_function = MLP(
            layers=[memory_size] + list(hidden_sizes) + [1],
            activation=activation,
            output_squeeze= True)

        self.rnn_out, self.rnn_out_last = None, None

    def forward(self, x, mask, h, a=None, horizon_t=1):
        x, h = self.feature_base(x, mask, h, horizon_t)
        a, logp_a, ent = self.policy(x, a)
        v = self.value_function(x).squeeze(dim=-1)
        return a, logp_a, ent, v, h.squeeze()

    def process_feature(self, x, mask, h, horizon_t=1):
        self.rnn_out, self.rnn_out_last = self.feature_base(x, mask, h, horizon_t)
        return self.rnn_out, self.rnn_out_last
