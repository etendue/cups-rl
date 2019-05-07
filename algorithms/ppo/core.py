import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import scipy.signal


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
        ent = policy.entropy().squeeze()
        return a, logp_a, ent


class ActorCritic(nn.Module):
    def __init__(self,
                 in_features,
                 action_space,
                 hidden_sizes=(64, 64),
                 memory_size = 128,
                 activation=torch.tanh,
                 output_activation=None):
        super(ActorCritic, self).__init__()

        self.feature_base = CNNRNNBase(
            input_shape=in_features,
            output_size=memory_size
        )
        # add last action as input
        self.action_n = action_space.n
        self.base_out_size = action_space.n + memory_size

        self.policy = CategoricalPolicy(
                self.base_out_size,
                hidden_sizes,
                activation,
                output_activation,
                action_dim=action_space.n)

        self.value_function = MLP(
            layers=[self.base_out_size] + list(hidden_sizes) + [1],
            activation=activation,
            output_squeeze=True)

        self.rnn_out, self.rnn_out_last = None, None

    def forward(self, x, mask, h_in, last_a=None, a=None, horizon_t=1):
        x, h_out = self.feature_base(x, mask, h_in, horizon_t)
        last_a_one_hot = torch.zeros(x.shape[0],self.action_n,device=x.device)
        if last_a:
            last_a_one_hot.scatter_(dim=1,last_a.long().unsqueeze(-1),1.0)
            last_a_one_hot = last_a_one_hot * mask
        x = torch.cat((x,last_a_one_hot),dim=1)
        a, logp_a, ent = self.policy(x, a)
        v = self.value_function(x).squeeze()
        return a, logp_a, ent, v, h_out.squeeze()

    def process_feature(self, x, mask, h, horizon_t=1):
        self.rnn_out, self.rnn_out_last = self.feature_base(x, mask, h, horizon_t)
        return self.rnn_out, self.rnn_out_last


class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, size, num_envs, memory_size, gamma=0.99, lam=0.95):
        self.obs_buf = torch.zeros((size, *obs_dim), dtype=torch.float32).cuda()
        self.act_buf = torch.zeros(size, dtype=torch.long).cuda()
        self.adv_buf = torch.zeros(size, dtype=torch.float32).cuda()
        self.rew_buf = torch.zeros(size, dtype=torch.float32).cuda()
        self.ret_buf = torch.zeros(size, dtype=torch.float32).cuda()
        self.val_buf = torch.zeros(size, dtype=torch.float32).cuda()
        self.logp_buf = torch.zeros(size, dtype=torch.float32).cuda()
        self.h_buf = torch.zeros((size, memory_size), dtype=torch.float32).cuda()
        self.mask_buf = torch.zeros(size, dtype=torch.float32).cuda()

        # to control the indexing
        self.ptr = torch.zeros(num_envs,dtype=torch.int).cuda()
        self.path_start_idx = torch.zeros(num_envs,dtype=torch.int).cuda()

        # constants
        self.gamma, self.lam, self.max_size, self.block_size = gamma, lam, size, size//num_envs

    def share_memory(self):
        self.obs_buf.share_memory_()
        self.act_buf.share_memory_()
        self.adv_buf.share_memory_()
        self.rew_buf.share_memory_()
        self.ret_buf.share_memory_()
        self.val_buf.share_memory_()
        self.logp_buf.share_memory_()
        self.h_buf.share_memory_()
        self.mask_buf.share_memory_()
        self.ptr.share_memory_()
        self.path_start_idx.share_memory_()

    def store(self, envid, obs, act, rew, val, logp, h, mask):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr[envid].item()  < self.block_size  # buffer has to have room so you can store
        ptr = self.ptr[envid].item()+ envid * self.block_size
        self.obs_buf[ptr].copy_(obs)
        self.act_buf[ptr].copy_(act)
        self.rew_buf[ptr].copy_(rew)
        self.val_buf[ptr].copy_(val)
        self.logp_buf[ptr].copy_(logp)
        self.h_buf[ptr].copy_(h)
        self.mask_buf[ptr].copy_(mask)
        self.ptr[envid] += 1

    def finish_path(self, envid, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """
        # map the index from environment block to whole buffer
        path_start_idx = self.path_start_idx[envid].item() + envid * self.block_size
        ptr = self.ptr[envid].item() + envid * self.block_size
        path_slice = slice(path_start_idx, ptr)

        last_v = torch.Tensor([last_val]).cuda()
        rews = torch.cat((self.rew_buf[path_slice], last_v), dim=0)
        vals = torch.cat((self.val_buf[path_slice], last_v), dim=0)
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = self._discount_cumsum(deltas, self.gamma * self.lam)
        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = self._discount_cumsum(rews, self.gamma)[:-1]
        self.path_start_idx[envid] = self.ptr[envid]

    def normalize_adv(self, mean_std=None, epsilon= 0.0001):
        """
        normalize the advantage with mean and standard deviation. If mean_std is not given, it calculate from date
        :param mean_std:
        :return: None
        """
        if mean_std == None:
            mean = self.adv_buf.mean()
            std = self.adv_buf.std()
        else:
            mean= mean_std[0]
            std = mean_std[1]
        self.adv_buf = (self.adv_buf - mean)/(std +epsilon)

    def batch_generator(self, batch_size):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer. Also, resets some pointers in the buffer.
        """
        if self.ptr.sum().item() != 0:
            assert self.ptr.sum().item() == self.max_size  # buffer has to be full before you can get
            self.ptr.copy_(torch.zeros_like(self.ptr))
            self.path_start_idx.copy_(torch.zeros_like(self.path_start_idx))
        last_a = torch.cat((torch.Tensor([0]).cuda(), self.act_buf[:-1]),dim=0)
        batch_sampler = BatchSampler( SubsetRandomSampler(range(self.max_size)), batch_size, drop_last=False)
        for idx in batch_sampler:
            yield [
                self.obs_buf[idx], self.act_buf[idx], self.adv_buf[idx], self.ret_buf[idx],
                self.logp_buf[idx], self.h_buf[idx], self.mask_buf[idx], last_a[idx]
            ]

    def _discount_cumsum(self, x, discount):
        """
        magic from rllab for computing discounted cumulative sums of vectors.
        input: vector x,[x0,x1,x2]
        output: [x0 + discount * x1 + discount^2 * x2, x1 + discount * x2, x2]
        """
        flipped_x = torch.flip(x,dims=(0,)).cpu()
        out = scipy.signal.lfilter([1], [1, float(-discount)], flipped_x, axis=0)
        t = torch.from_numpy(out).cuda()
        return torch.flip(t, dims=(0,))