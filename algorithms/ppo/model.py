import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import numpy as np


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


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)


class ActorCritic(torch.nn.Module):
    """
    Mainly Ikostrikov's implementation of A3C (https://arxiv.org/abs/1602.01783).

    Processes an input image (with num_input_channels) with 4 conv layers,
    interspersed with 4 elu activation functions. The output of the final layer is then flattened
    and passed to an LSTM (with previous or initial hidden and cell states (hx and cx)).
    The new hidden state is used as an input to the critic and value nn.Linear layer heads,
    The final output is then predicted value, action logits, hx and cx.
    """

    def __init__(self, num_input_channels, num_outputs, frame_dim):
        super(ActorCritic, self).__init__()
        self.conv1 = nn.Conv2d(num_input_channels, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)

        # assumes square image
        self.rnn_input_size = calculate_output_size_after_4_conv_layers(frame_dim)
        self.rnn_state_size = 256
        self.lstm = nn.GRUCell(self.rnn_input_size, self.rnn_state_size)  # for 128x128 input

        self.critic_linear = nn.Linear(256, 1)
        self.actor_linear = nn.Linear(256, num_outputs)

        self.apply(weights_init)
        self.actor_linear.weight.data = normalized_columns_initializer(
                                            self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)
        self.critic_linear.weight.data = normalized_columns_initializer(
                                            self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)

        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)

        self.train()

    def forward(self, inputs, actions=None):

        outputs, rnn_out = self._feature_extractor(inputs)
        v = self.critic_linear(outputs)
        logits = self.actor_linear(outputs)
        dist = Categorical(logits=logits)

        if actions is None:
            actions = dist.sample()

        logp_a = dist.log_prob(actions)
        entropy = dist.entropy()

        v = v.squeeze()
        a = actions.squeeze()
        logp_a = logp_a.squeeze()
        ent = entropy.squeeze()
        rnn_out = rnn_out.squeeze()

        return v, a, logp_a, ent, rnn_out

    def value_function(self,inputs):
        outputs, _ = self._feature_extractor(inputs)
        v = self.critic_linear(outputs)
        return v.squeeze()

    def _feature_extractor(self,inputs):

        # inputs has shape [T, batch,channel, w, h]
        # rnn_in has shape [batch, hidden_state_size]
        # mask has shape   [T, batch,1]
        inputs, rnn_in, mask = inputs

        assert (len(inputs.size()) == 3 and len(rnn_in.size()) == 1 and len(mask.size()) == 1) or \
               (len(inputs.size()) == 5 and len(rnn_in.size()) == 2 and len(mask.size()) == 3)

        if len(inputs.size()) == 3:  # if batch forgotten, with 1 time step
            x = inputs.unsqueeze(0).unsqueeze(0)
            rnn_in.unsqueeze(0)
            mask = mask.unsqueeze(0).unsqueeze(0)

        # data is organized in order [T, batch, channel, w,h]
        T, batch_size, c, w, h = inputs.size()

        # flatten the data to go through the CNN network
        x = x.view(T * batch_size, c, w, h)

        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))

        # convert back to time sequence for RNN cell
        x = x.view(T, batch_size, self.rnn_input_size)
        outputs = torch.empty(T, batch_size, self.rnn_state_size, device = x.device())

        hidden_state = rnn_in
        for t in range(T):
            out = hidden_state= self.gru(x[t], hidden_state* mask[t])
            outputs[t].copy_(out)

        outputs = outputs.view(T * batch_size, self.rnn_state_size)
        # the output has shape [T*batch, lstm_cell_size], hx_out, cx_out [batch, lstm_cell_size]
        return outputs, hidden_state
