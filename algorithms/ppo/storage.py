import torch
from math import ceil,floor
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

def _flatten_helper(T, N, _tensor):
    return _tensor.view(T * N, *_tensor.size()[2:])


class RolloutStorage(object):
    def __init__(self, num_steps, num_processes, obs_shape, action_space, state_size):
        self.observations = torch.zeros(num_steps + 1, num_processes, *obs_shape)
        self.states = torch.zeros(num_steps + 1, num_processes, state_size)
        self.rewards = torch.zeros(num_steps, num_processes, 1)
        self.value_preds = torch.zeros(num_steps + 1, num_processes, 1)
        self.returns = torch.zeros(num_steps + 1, num_processes, 1)
        self.advantages = torch.zeros(num_steps, num_processes, 1)
        self.action_log_probs = torch.zeros(num_steps, num_processes, 1)
        if action_space.__class__.__name__ == 'Discrete':
            action_shape = 1
        else:
            action_shape = action_space.shape[0]
        self.actions = torch.zeros(num_steps, num_processes, action_shape)
        if action_space.__class__.__name__ == 'Discrete':
            self.actions = self.actions.long()

        self.num_steps = num_steps
        self.step = 0

    def cuda(self):
        self.observations = self.observations.cuda()
        self.states = self.states.cuda()
        self.rewards = self.rewards.cuda()
        self.value_preds = self.value_preds.cuda()
        self.returns = self.returns.cuda()
        self.action_log_probs = self.action_log_probs.cuda()
        self.actions = self.actions.cuda()
        self.masks = self.masks.cuda()
        self.time_steps= self.time_steps.cuda()

    def insert(self, current_obs, state, action, action_log_prob, value_pred, reward, mask):
        self.observations[self.step + 1].copy_(current_obs)
        self.states[self.step + 1].copy_(state)
        self.actions[self.step].copy_(action)
        self.action_log_probs[self.step].copy_(action_log_prob)
        self.value_preds[self.step].copy_(value_pred)
        self.rewards[self.step].copy_(reward)
        self.masks[self.step + 1].copy_(mask)

        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        self.observations[0].copy_(self.observations[-1])
        self.states[0].copy_(self.states[-1])
        self.masks[0].copy_(self.masks[-1])

    def compute_returns(self, next_value, use_gae, gamma, tau):
        if use_gae:
            self.value_preds[-1] = next_value
            gae = 0
            for step in reversed(range(self.rewards.size(0))):
                delta = self.rewards[step] + gamma * self.value_preds[step + 1] * self.masks[step + 1] - self.value_preds[step]
                gae = delta + gamma * tau * self.masks[step + 1] * gae
                self.returns[step] = gae + self.value_preds[step]
        else:
            self.returns[-1] = next_value
            for step in reversed(range(self.rewards.size(0))):
                self.returns[step] = self.returns[step + 1] * \
                    gamma * self.masks[step + 1] + self.rewards[step]

        advantages = self.returns[:-1] - self.value_preds[:-1]
        self.advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

    def sample_generator(self,num_mini_batch,horizon_length):
        t = horizon_length
        n = self.observations.shape[0]
        assert n%t == 0, f"total steps ({n}) is multitude of horizon length {t}"

        o = self.observations[:-1].view(-1,t,*self.observations.shape[2:])
        s = self.states[:-1].view(-1,t,*self.states.shape[2:])
        a = self.actions.view(-1,t,*self.actions.shape[2:])
        r = self.returns.view(-1,t, *self.returns.shape[2:])
        m = self.masks.view(-1,t, *self.masks.shape[2:])

        old_logp = self.action_log_probs.view(-1, t,*self.action_log_probs.shape[2:])
        adv = self.advantages.view(-1,t, *self.advantages.shape[2:])

        num_slices = o.shape[0]
        mini_batch_size = num_slices//num_mini_batch
        sampler = BatchSampler(SubsetRandomSampler(range(num_slices)), mini_batch_size, drop_last=True)

        for idx in sampler:
            yield o[idx].view(-1,*o.shape[2:]),\
                  s[idx].view(-1,*s.shape[2:]),\
                  a[idx].view(-1,*a.shape[2:]),\
                  r[idx].view(-1,*r.shape[2:]),\
                  m[idx].view(-1,*m.shape[2:]),\
                  old_logp[idx].view(-1,old_logp.shape[2:]),\
                  adv[idx].view(-1,adv.shape[2:])
