import torch
from math import ceil,floor
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


class RolloutStorage(object):
    def __init__(self, num_steps, num_processes, obs_shape, state_size):
        self.obs = torch.zeros(num_steps + 1, num_processes, *obs_shape)
        self.memory = torch.zeros(num_steps + 1, num_processes, state_size)
        self.masks = torch.ones(num_steps + 1, num_processes, 1)
        self.v = torch.zeros(num_steps + 1, num_processes, 1)
        self.g = torch.zeros(num_steps + 1, num_processes, 1)

        #
        self.adv = torch.zeros(num_steps, num_processes, 1)
        self.r = torch.zeros(num_steps, num_processes, 1)
        self.logp_a = torch.zeros(num_steps, num_processes, 1)
        self.a = torch.zeros(num_steps, num_processes, 1).long()

        self.num_steps = num_steps
        self.index = 0

    def cuda(self):
        self.obs = self.obs.cuda()
        self.memory = self.memory.cuda()
        self.r = self.r.cuda()
        self.v = self.v.cuda()
        self.g = self.g.cuda()
        self.logp_a = self.logp_a.cuda()
        self.a = self.a.cuda()

    def insert(self, next_obs, rnn_out, action, action_log_prob, value, reward, mask):
        idx = self.index
        self.obs[idx + 1].copy_(next_obs)
        self.memory[idx + 1].copy_(rnn_out)
        self.a[idx].copy_(action)
        self.logp_a[idx].copy_(action_log_prob)
        self.v[idx].copy_(value)
        self.r[idx].copy_(reward)
        self.masks[idx + 1].copy_(mask)

        self.index = (self.index + 1) % self.num_steps

    def after_update(self):
        self.obs[0].copy_(self.obs[-1])
        self.memory[0].copy_(self.memory[-1])
        self.masks[0].copy_(self.masks[-1])

    def compute_returns(self, next_value, use_gae, gamma, tau):
        if use_gae:
            self.v[-1] = next_value
            gae = 0
            for step in reversed(range(self.r.size(0))):
                delta = self.r[step] + gamma * self.v[step + 1] * self.masks[step + 1] - self.v[step]
                gae = delta + gamma * tau * self.masks[step + 1] * gae
                self.g[step] = gae + self.v[step]
        else:
            self.g[-1] = next_value
            for step in reversed(range(self.r.size(0))):
                self.g[step] = self.g[step + 1] * \
                               gamma * self.masks[step + 1] + self.r[step]

        advantages = self.g[:-1] - self.v[:-1]
        self.adv = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

    def sample_generator(self,num_mini_batch, horizon_length=32):
        t = horizon_length
        n = self.obs.shape[0]
        assert n % t == 0, f"total steps ({n}) is multitude of horizon length {t}"

        o = self.obs[:-1].view(t, -1, *self.obs.shape[2:])
        h = self.memory[:-1].view(t, -1, *self.memory.shape[2:])
        m = self.masks[:-1].view(t,-1, *self.masks.shape[2:])
        g = self.g[:-1].view(t, -1, *self.g.shape[2:])

        a = self.a.view(t, -1, *self.a.shape[2:])
        logp_a = self.logp_a.view(t, -1, *self.logp_a.shape[2:])
        adv = self.adv.view(t, -1, *self.adv.shape[2:])

        num_slices = o.shape[0]
        mini_batch_size = num_slices//num_mini_batch
        sampler = BatchSampler(SubsetRandomSampler(range(num_slices)), mini_batch_size, drop_last=True)

        for idx in sampler:
            yield o[:,idx], h[0,idx].squeeze(0), a[:,idx], g[:,idx], m[:,idx], logp_a[:,idx], adv[:,idx]
