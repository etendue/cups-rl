import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from mpi4py import MPI
import subprocess
import sys
import os


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
        # TODO: initialization weights and bias
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
            a = policy.sample()
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

        self.feature_base_ = CNNRNNBase(
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
            output_squeeze=True)

        self.rnn_out, self.rnn_out_last = None, None

    def forward(self, x, mask, h, a=None, horizon_t=1):
        x, h = self.feature_base_(x, mask, h, horizon_t)
        a, logp_a, ent = self.policy(x, a)
        v = self.value_function(x)
        return a, logp_a, ent, v, h

    def process_feature(self, x, mask, h, horizon_t=1):
        self.rnn_out, self.rnn_out_last = self.feature_base_(x, mask, h, horizon_t)
        return self.rnn_out, self.rnn_out_last


def mpi_fork(n, bind_to_core=False):
    """
    Re-launches the current script with workers linked by MPI.

    Also, terminates the original process that launched it.

    Taken almost without modification from the Baselines function of the
    `same name`_.

    .. _`same name`: https://github.com/openai/baselines/blob/master/baselines/common/mpi_fork.py

    Args:
        n (int): Number of process to split into.

        bind_to_core (bool): Bind each MPI process to a core.
    """
    if n<=1:
        return
    if os.getenv("IN_MPI") is None:
        env = os.environ.copy()
        env.update(
            MKL_NUM_THREADS="1",
            OMP_NUM_THREADS="1",
            IN_MPI="1"
        )
        args = ["mpirun", "-np", str(n)]
        if bind_to_core:
            args += ["-bind-to", "core"]
        args += [sys.executable] + sys.argv
        subprocess.check_call(args, env=env)
        sys.exit()


def msg(m, string=''):
    print(('Message from %d: %s \t '%(MPI.COMM_WORLD.Get_rank(), string))+str(m))


def proc_id():
    """Get rank of calling process."""
    return MPI.COMM_WORLD.Get_rank()


def allreduce(*args, **kwargs):
    return MPI.COMM_WORLD.Allreduce(*args, **kwargs)


def num_procs():
    """Count active MPI processes."""
    return MPI.COMM_WORLD.Get_size()


def broadcast(x, root=0):
    MPI.COMM_WORLD.Bcast(x, root=root)


def mpi_op(x, op):
    x, scalar = ([x], True) if np.isscalar(x) else (x, False)
    x = np.asarray(x, dtype=np.float32)
    buff = np.zeros_like(x, dtype=np.float32)
    allreduce(x, buff, op=op)
    return buff[0] if scalar else buff


def mpi_sum(x):
    return mpi_op(x, MPI.SUM)


def mpi_avg(x):
    """Average a scalar or vector over MPI processes."""
    return mpi_sum(x) / num_procs()


def mpi_statistics_scalar(x, with_min_and_max=False):
    """
    Get mean/std and optional min/max of scalar x across MPI processes.

    Args:
        x: An array containing samples of the scalar to produce statistics
            for.

        with_min_and_max (bool): If true, return min and max of x in
            addition to mean and std.
    """
    x = np.array(x, dtype=np.float32)
    global_sum, global_n = mpi_sum([np.sum(x), len(x)])
    mean = global_sum / global_n

    global_sum_sq = mpi_sum(np.sum((x - mean)**2))
    std = np.sqrt(global_sum_sq / global_n)  # compute global std

    if with_min_and_max:
        global_min = mpi_op(np.min(x) if len(x) > 0 else np.inf, op=MPI.MIN)
        global_max = mpi_op(np.max(x) if len(x) > 0 else -np.inf, op=MPI.MAX)
        return mean, std, global_min, global_max
    return mean, std


def sync_all_params(param, root=0):
    data = torch.nn.utils.parameters_to_vector(param).detach().numpy()
    broadcast(data, root)
    torch.nn.utils.vector_to_parameters(torch.from_numpy(data), param)


def average_gradients(param_groups):
    for param_group in param_groups:
        for p in param_group['params']:
            if p.requires_grad:
                p.grad.data.copy_(torch.Tensor(mpi_avg(p.grad.data.numpy())))
