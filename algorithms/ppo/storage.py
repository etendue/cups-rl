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
        self.action_log_probs = torch.zeros(num_steps, num_processes, 1)
        if action_space.__class__.__name__ == 'Discrete':
            action_shape = 1
        else:
            action_shape = action_space.shape[0]
        self.actions = torch.zeros(num_steps, num_processes, action_shape)
        if action_space.__class__.__name__ == 'Discrete':
            self.actions = self.actions.long()
        self.masks = torch.ones(num_steps + 1, num_processes, 1)

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


    def feed_forward_generator(self, advantages, num_mini_batch):
        num_steps, num_processes = self.rewards.size()[0:2]
        batch_size = num_processes * num_steps
        assert batch_size >= num_mini_batch, (
            f"PPO requires the number processes ({num_processes}) "
            f"* number of steps ({num_steps}) = {num_processes * num_steps} "
            f"to be greater than or equal to the number of PPO mini batches ({num_mini_batch}).")
        mini_batch_size = batch_size // num_mini_batch
        sampler = BatchSampler(SubsetRandomSampler(range(batch_size)), mini_batch_size, drop_last=False)
        for indices in sampler:
            observations_batch = self.observations[:-1].view(-1,
                                        *self.observations.size()[2:])[indices]
            states_batch = self.states[:-1].view(-1, self.states.size(-1))[indices]
            actions_batch = self.actions.view(-1, self.actions.size(-1))[indices]
            return_batch = self.returns[:-1].view(-1, 1)[indices]
            masks_batch = self.masks[:-1].view(-1, 1)[indices]
            old_action_log_probs_batch = self.action_log_probs.view(-1, 1)[indices]
            adv_targ = advantages.view(-1, 1)[indices]

            yield observations_batch, states_batch, actions_batch, \
                return_batch, masks_batch, old_action_log_probs_batch, adv_targ

    def recurrent_generator(self, advantages, num_mini_batch,num_time_step = 1):
        num_processes = self.rewards.size(1)
        num_t_slices = floor(self.num_steps*num_processes/num_time_step)
        mini_batch_size = num_t_slices//num_mini_batch
        sampler = BatchSampler(SubsetRandomSampler(range(num_t_slices)), mini_batch_size, drop_last=True)
        T, N = self.num_steps, num_processes

        #stack the tensor along the dim of processes
        observations_flatted= self.observations[:-1].view(T*N, *self.observations.shape[2:])
        states_flatted = self.states[:-1].view(T*N, *self.states.shape[2:])
        actions_flatted = self.actions.view(T*N, *self.actions.shape[2:])
        return_flatted = self.returns[:-1].view(T*N,*self.returns.shape[2:])
        masks_flatted = self.masks[:-1].view(T*N,*self.masks.shape[2:])
        old_action_log_probs_flatted = self.action_log_probs.view(T*N,*self.action_log_probs.shape[2:])
        advantages_flatted = advantages.view(T*N,*advantages.shape[2:])

        num_of_data = observations_flatted.shape[0]
        for indices in sampler:
            batch_idx = []
            state_idx = []
            for i in indices:
                t = i * num_time_step
                state_idx.append(t)
                batch_idx.extend(list(range(t,t+num_time_step)))

            observations_batch = observations_flatted[batch_idx]
            states_batch = states_flatted[state_idx]
            actions_batch = actions_flatted[batch_idx]
            return_batch = return_flatted[batch_idx]
            masks_batch = masks_flatted[batch_idx]
            old_action_log_probs_batch = old_action_log_probs_flatted[batch_idx]
            adv_targ = advantages_flatted[batch_idx]
            ct_batch = torch.Tensor(batch_idx, dtype = torch.int32)
            ct_batch.data_ = ct_batch.data % T

            yield observations_batch, states_batch, actions_batch, \
              return_batch, masks_batch, old_action_log_probs_batch, adv_targ, ct_batch
