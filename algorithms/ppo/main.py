import copy
import os
import time

import numpy as np
import torch
from tensorboardX import SummaryWriter
import argparse

from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.vec_normalize import VecNormalize
from gym_ai2thor.envs.ai2thor_env import  AI2ThorEnv
from algorithms.ppo.model import ActorCritic
from algorithms.ppo.storage import RolloutStorage
from algorithms.ppo.ppo import PPO


def main(args):
    steps_per_update = args.steps_per_epoch
    local_steps_per_update = steps_per_update // args.num_processes
    total_updates = int(args.num_frames) // steps_per_update

    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    if args.save_dir != "":
        save_path = os.path.join(args.save_dir, args.algo)
        try:
            os.makedirs(save_path)
        except OSError:
            print(f"Can not create dir {save_path}, model can not be saved")
            args.save_dir = ""

    writer = SummaryWriter(comment="ppo_ai2thor")
    envs = [AI2ThorEnv for _ in range(args.num_processes)]

    if args.num_processes > 1:
        envs = SubprocVecEnv(envs)
    else:
        envs = DummyVecEnv(envs)

    if len(envs.observation_space.shape) == 1:
        envs = VecNormalize(envs, gamma=args.gamma)

    num_channel,w_dim, h_dim = envs.observation_space.shape
    action_dim = envs.action_space.n
    actor_critic = ActorCritic(num_channel,action_dim,w_dim)

    # load saved model for continuous training
    if args.saved_model:
        actor_critic.load_state_dict(torch.load(args.saved_model).state_dict())

    if args.cuda:
        actor_critic.cuda()

    agent = PPO(actor_critic,
                args.clip_param,
                args.ppo_epoch,
                args.num_mini_batch,
                args.value_loss_coef,
                args.entropy_coef,
                lr=args.lr,
                eps=args.eps,
                max_grad_norm=args.max_grad_norm)

    buf = RolloutStorage(local_steps_per_update,
                         args.num_processes,
                         envs.observation_space.shape,
                         actor_critic.rnn_state_size)

    # These variables are used to compute average rewards for all processes.
    episode_rewards = np.zeros([args.num_processes])
    final_rewards = np.zeros([args.num_processes])

    if args.cuda:
        buf.cuda()

    obs0 = envs.reset()/255
    buf.obs[0].copy_(torch.tensor(obs0).unsqueeze(1))

    start = time.time()
    for ep in range(total_updates):
        epsisode_start = time.time()
        actor_critic.eval()
        for step in range(local_steps_per_update):
            # Sample actions
            state = (buf.obs[step], buf.memory[step], buf.masks[step])
            v, pi, log_pi, _, h_rnn = actor_critic(state)
            a = pi.cpu().numpy()

            # interact with environment, Tricky thing here is the envs reset itsself when one env is done
            # so next_o is for next episode
            next_o, r, done, _ = envs.step(a)
            r = np.array(r).astype(float)

            # If done then clean the history of observations.
            episode_rewards += r
            done = np.array(done).astype(float)
            final_rewards *= done
            final_rewards += (1 - done) * episode_rewards
            episode_rewards *= done

            # convert to Torch tensor
            next_o = torch.Tensor(next_o/255)
            r = torch.Tensor(r)
            mask = torch.Tensor(done)

            buf.insert(next_o, h_rnn, pi, log_pi, v, r, mask)

        exp_collect_stop = time.time()
        next_value = actor_critic.value_function((buf.obs[-1], buf.memory[-1], buf.masks[-1]))
        buf.compute_returns(next_value, args.use_gae, args.gamma, args.tau)

        actor_critic.train()
        # do PPO clip_param decay
        agent.clip_param = (args.clip_param - 0.001) * (total_updates - ep)/total_updates + 0.001
        value_loss, action_loss, dist_entropy = agent.update(buf)
        buf.after_update()

        if ep % args.save_interval == 0 and args.save_dir != "":
            # A really ugly way to save a model to CPU
            save_model = actor_critic
            if args.cuda:
                save_model = copy.deepcopy(actor_critic).cpu()
            torch.save(save_model, os.path.join(save_path, args.env_name + ".pt"))

        if ep % args.log_interval == 0:
            end = time.time()
            total_num_steps = (ep + 1) * args.num_processes * args.num_steps
            print("Updates {}, num timesteps {}, FPS {}, mean/median reward {:.1f}/{:.1f},"
                  " min/max reward {:.1f}/{:.1f}, entropy {:.5f}, value loss {:.5f}, policy loss {:.5f}".
                  format(ep, total_num_steps,
                         int(total_num_steps / (end - start)),
                         final_rewards.mean(),
                         final_rewards.median(),
                         final_rewards.min(),
                         final_rewards.max(), dist_entropy,
                         value_loss, action_loss))

            episode_t = end - epsisode_start
            exp_per = int(100*(exp_collect_stop - epsisode_start)/episode_t)
            print(f"Episode Time {episode_t}, experience collection:{exp_per} % for training {1 -exp_per}")

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
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument('--lr', type=float, default=2.5e-4,
                        help='learning rate (default: 7e-4)')
    parser.add_argument('--eps', type=float, default=1e-5,
                        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument('--alpha', type=float, default=0.99,
                        help='RMSprop optimizer apha (default: 0.99)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='discount factor for rewards (default: 0.99)')
    parser.add_argument('--use-gae', action='store_true', default=True,
                        help='use generalized advantage estimation')
    parser.add_argument('--tau', type=float, default=0.95,
                        help='gae parameter (default: 0.95)')
    parser.add_argument('--entropy-coef', type=float, default=0.01,
                        help='entropy term coefficient (default: 0.01)')
    parser.add_argument('--value-loss-coef', type=float, default=0.01,
                        help='value loss coefficient (default: 0.5)')
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                        help='max norm of gradients (default: 0.5)')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--num-processes', type=int, default=2,
                        help='how many training CPU processes to use (default: 2)')
    parser.add_argument('--steps-per-epoch', type=int, default=1024,
                        help='steps to go before optimize the network')
    parser.add_argument('--ppo-epoch', type=int, default=4,
                        help='number of ppo epochs (default: 4)')
    parser.add_argument('--num-mini-batch', type=int, default=8,
                        help='number of batches for ppo (default: 8)')
    parser.add_argument('--clip-param', type=float, default=0.2,
                        help='ppo clip parameter (default: 0.2)')
    parser.add_argument('--log-interval', type=int, default=1,
                        help='log interval, one log per n updates (default: 10)')
    parser.add_argument('--save-interval', type=int, default=100,
                        help='save interval, one save per n updates (default: 10)')
    parser.add_argument('--num-frames', type=int, default=10e6,
                        help='number of frames to train (default: 10e6)')
    parser.add_argument('--save-dir', default='',
                        help='directory to save agent logs (default: )')
    parser.add_argument('--saved-model', default=None,
                        help='model to load from previous saved one')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')

    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    main(args)
