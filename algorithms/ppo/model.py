import torch
import torch.nn as nn
import torch.nn.functional as F
from algorithms.ppo.distributions import Categorical
from algorithms.ppo.utils import init


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Policy(nn.Module):
    def __init__(self, obs_shape, action_space, recurrent_policy):
        super(Policy, self).__init__()
        if len(obs_shape) == 3:
            self.base = CNNBase(obs_shape[0], recurrent_policy)
        else:
            raise NotImplementedError

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(self.base.output_size + self.time_emb_size, num_outputs)
        else:
            raise NotImplementedError

        self.state_size = self.base.state_size

    def forward(self, inputs, states, masks):
        raise NotImplementedError

    def act(self, inputs, states, masks, ct, deterministic=False):
        value, actor_features, states = self.base(inputs, states, masks, ct)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)

        return value, action, action_log_probs, states

    def get_value(self, inputs, states, masks, ct):
        value, _, _ = self.base(inputs, states, masks, ct)
        return value

    def evaluate_actions(self, inputs, states, masks, action, ct):
        value, actor_features, states = self.base(inputs, states, masks, ct)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, states


class CNNBase(nn.Module):
    def __init__(self, num_inputs, use_gru, episode_length):
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
            #init_(nn.Linear(32 * 7 * 7, 512)),
            # size does nt match
            init_(nn.Linear(32 * 12 * 12, 512)),
            nn.ReLU()
        )

        if use_gru:
            self.gru = nn.GRUCell(512, 512)
            nn.init.orthogonal_(self.gru.weight_ih.data)
            nn.init.orthogonal_(self.gru.weight_hh.data)
            self.gru.bias_ih.data.fill_(0)
            self.gru.bias_hh.data.fill_(0)

        init_ = lambda m: init(m,
          nn.init.orthogonal_,
          lambda x: nn.init.constant_(x, 0))

        self.time_emb_size = 32
        self.time_embedding = nn.Embedding(episode_length,self.time_emb_size)

        self.critic_linear = init_(nn.Linear(512+self.time_emb_size , 1))

        self.train()

    def forward(self, inputs, states, masks, ct):
        x = self.main(inputs / 255.0)

        if hasattr(self, 'gru'):
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

        time_emb = self.time_embedding(ct)

        return self.critic_linear(torch.cat(x, time_emb)), x, states
