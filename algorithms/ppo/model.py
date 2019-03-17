import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from algorithms.ppo.utils import init


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class CNN_LSTM_ActorCritic(nn.Module):
    def __init__(self, obs_shape, action_space, episode_length):
        super(CNN_LSTM_ActorCritic, self).__init__()

        self.base = CNNBase(obs_shape[0],episode_length)
        init_ = lambda m: init(m,
                               nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0))
        self.state_size = self.base.state_size
        num_outputs = action_space.n

        self.critic = init_(nn.Linear(512, 1))
        self.actor = init_(nn.Linear(self.base.output_size, num_outputs))
        self.train()

    def forward(self, inputs, states, masks):
        raise NotImplementedError

    def act(self, inputs, states, masks):
        x, states = self.base(inputs, states, masks)
        logits = self.actor(x)
        dist = Categorical(logits=logits)
        action = dist.sample()
        action_log_probs = dist.log_prob(action)
        value = self.critic(x)

        return value, action, action_log_probs, states

    def get_value(self, inputs, states, masks, ct):
        x, states = self.base(inputs, states, masks)
        return self.critic(x)

    def evaluate_actions(self, inputs, states, masks, action):
        x, states = self.base(inputs, states, masks)
        logits = self.actor(x)
        dist = Categorical(logits=logits)
        action_log_probs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        value =self.critic(x)

        return value, action_log_probs, dist_entropy, states


class CNNBase(nn.Module):
    def __init__(self, num_inputs, episode_length):
        super(CNNBase, self).__init__()

        init_ = lambda m: init(m,
                      nn.init.orthogonal_,
                      lambda x: nn.init.constant_(x, 0),
                      nn.init.calculate_gain('relu'))

        self.main = nn.Sequential(
            init_(nn.Conv2d(num_inputs, 32, 8, stride=4)),
            nn.ReLU(),
            init_(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            init_(nn.Conv2d(64, 32, 3, stride=1)),
            nn.ReLU(),
            Flatten(),
            init_(nn.Linear(32 * 12 * 12, 512)),
            nn.ReLU()
        )

        self.gru = nn.GRUCell(512, 512)
        nn.init.orthogonal_(self.gru.weight_ih.data)
        nn.init.orthogonal_(self.gru.weight_hh.data)
        self.gru.bias_ih.data.fill_(0)
        self.gru.bias_hh.data.fill_(0)
        self.train()

    @property
    def output_size(self):
        return 512

    @property
    def state_size(self):
        return 512


    def forward(self, inputs, states, masks):
        x = self.main(inputs / 255.0)
        if inputs.size(0) == states.size(0):
            x = states = self.gru(x, states * masks)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = states.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N, 1)

            outputs = []
            for i in range(T):
                hx = states = self.gru(x[i], states * masks[i])
                outputs.append(hx)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.stack(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)

        return  x, states
