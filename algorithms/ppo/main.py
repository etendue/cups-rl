import copy
import os
import time

import numpy as np
import torch
from tensorboardX import SummaryWriter

from algorithms.ppo.arguments import get_args
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.vec_normalize import VecNormalize
from gym_ai2thor.envs.ai2thor_env import  AI2ThorEnv
from algorithms.ppo.model import CNN_LSTM_ActorCritic
from algorithms.ppo.storage import RolloutStorage
from algorithms.ppo.ppo import PPO


def main():
    args = get_args()
    num_updates = int(args.num_frames) // args.num_steps // args.num_processes

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    if args.save_dir != "":
        save_path = os.path.join(args.save_dir, args.algo)
        try:
            os.makedirs(save_path)
        except OSError:
            print(f"Can not create dir {save_path}, model can not be saved")
            args.save_dir =""
            pass

    writer = SummaryWriter(comment="ppo_ai2thor")
    torch.set_num_threads(1)

    envs = [AI2ThorEnv() for _ in range(args.num_processes)]

    if args.num_processes > 1:
        envs = SubprocVecEnv(envs)
    else:
        envs = DummyVecEnv(envs)

    if len(envs.observation_space.shape) == 1:
        envs = VecNormalize(envs, gamma=args.gamma)

    obs_shape = envs.observation_space.shape
    actor_critic = CNN_LSTM_ActorCritic(obs_shape,
                                        envs.action_space,
                                        args.recurrent_policy,
                                        args.num_steps)

    # load saved model for continuous training
    if args.saved_model is not None:
        actor_critic.load_state_dict(torch.load(args.saved_model).state_dict())

    if args.cuda:
        actor_critic.cuda()

    agent = PPO(actor_critic,
                args.clip_param,
                args.ppo_epoch,
                args.num_mini_batch,
                args.num_time_step,
                args.value_loss_coef,
                args.entropy_coef,
                lr=args.lr,
                eps=args.eps,
                max_grad_norm=args.max_grad_norm)

    rollouts = RolloutStorage(args.num_steps,
                              args.num_processes,
                              obs_shape,
                              envs.action_space,
                              actor_critic.state_size)

    # These variables are used to compute average rewards for all processes.
    episode_rewards = torch.zeros([args.num_processes, 1])
    final_rewards = torch.zeros([args.num_processes, 1])

    if args.cuda:
        rollouts.cuda()

    obs0 = envs.reset()
    rollouts.observations[0].copy_(torch.Tensor(obs0 / 255))

    start = time.time()
    for j in range(num_updates):
        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, states = actor_critic.act(
                        rollouts.observations[step],
                        rollouts.hidden_state[step],
                        rollouts.masks[step])
            cpu_actions = action.squeeze(1).cpu().numpy()

            # Obser reward and next obs
            next_o, r, done, _ = envs.step(cpu_actions)
            r = torch.from_numpy(np.expand_dims(np.stack(r), 1)).float()
            episode_rewards += r

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            final_rewards *= masks
            final_rewards += (1 - masks) * episode_rewards
            episode_rewards *= masks

            next_o = torch.FloatTensor(next_o/255)
            next_o *= masks

            rollouts.insert(next_o, states, action, action_log_prob, value, r, masks)


        with torch.no_grad():
            next_value = actor_critic.get_value(rollouts.observations[-1],
                                                rollouts.hidden_state[-1],
                                                rollouts.masks[-1]).detach()

        rollouts.compute_returns(next_value, args.use_gae, args.gamma, args.tau)
        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()

        if j % args.save_interval == 0 and args.save_dir != "":
            # A really ugly way to save a model to CPU
            save_model = actor_critic
            if args.cuda:
                save_model = copy.deepcopy(actor_critic).cpu()
            torch.save(save_model, os.path.join(save_path, args.env_name + ".pt"))

        if j % args.log_interval == 0:
            end = time.time()
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            print("Updates {}, num timesteps {}, FPS {}, mean/median reward {:.1f}/{:.1f},"
                  " min/max reward {:.1f}/{:.1f}, entropy {:.5f}, value loss {:.5f}, policy loss {:.5f}".
                  format(j, total_num_steps,
                         int(total_num_steps / (end - start)),
                         final_rewards.mean(),
                         final_rewards.median(),
                         final_rewards.min(),
                         final_rewards.max(), dist_entropy,
                         value_loss, action_loss))

            writer.add_scalars("Reward", {"mean": final_rewards.mean(),
                                          "median": final_rewards.median(),
                                          "min": final_rewards.min(),
                                          "max": final_rewards.max()
                                          },
                               total_num_steps)
            writer.add_scalars("Losses",
                               {"value": value_loss, "policy": action_loss},
                               total_num_steps)
            writer.add_scalar("Entropy", dist_entropy, total_num_steps)


if __name__ == "__main__":
    main()
