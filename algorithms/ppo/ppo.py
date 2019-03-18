import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class PPO(object):
    def __init__(self,
                 actor_critic,
                 clip_param,
                 ppo_epoch,
                 num_mini_batch,
                 value_loss_coef,
                 entropy_coef,
                 lr=None,
                 eps=None,
                 max_grad_norm=None,
                 target_kl=0.01):

        self.actor_critic = actor_critic

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)
        self.target_kl = target_kl

    def update(self, rollouts):

        value_loss_epoch = 0
        action_loss_epoch = 0
        entropy_epoch = 0

        num_updates = 0

        for e in range(self.ppo_epoch):
            data_generator = rollouts.batch_generator(self.num_mini_batch)

            for sample in data_generator:
                o, s, a, g, m, old_logp, adv_targ = sample

                # Reshape to do in a single forward pass for all steps
                v, logp, ent, _ = self.actor_critic(o, s, m, a)

                ratio = torch.exp(logp - old_logp)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                           1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(g, v)
                ent_bonus = ent.mean()

                self.optimizer.zero_grad()
                total_loss = value_loss * self.value_loss_coef + action_loss - ent_bonus * self.entropy_coef
                total_loss.backward()
                # nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                entropy_epoch += ent_bonus.item()
                num_updates += 1
                with torch.no_grad():
                    _, logp, _, _ = self.actor_critic(o, s, m, a)
                    kl = (old_logp - logp).mean()
                    if kl > 1.5 * self.target_kl:
                        print(f'!!!! step {num_updates} reaches max kl {1.5*self.target_kl}.')

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        entropy_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, entropy_epoch
