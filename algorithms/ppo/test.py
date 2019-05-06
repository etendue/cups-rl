from  gym_ai2thor.envs.ai2thor_env import AI2ThorEnv
import numpy as np
import torch
from algorithms.ppo.core import ActorCritic
import sys


def reset(env, rnn_size):
    o, d, r = env.reset() / 255., False, 0.
    mask_t = torch.Tensor([1.]).cuda()
    o_t = torch.Tensor(o).cuda().unsqueeze(dim=0)  # 128x128 -> 1x128x128
    h_t = torch.zeros(rnn_size).cuda()
    r_t = torch.Tensor([r]).cuda()
    return o_t, r_t, h_t, mask_t, d

def tester(env, model, rnn_size, n = 5):
    env = AI2ThorEnv(config_file="config_files/OneMugTest.json")
    episode_reward = []
    for _ in range(n):
        # Wait for trainer to inform next job
        total_r = 0.
        o_t, r_t, h_t, mask_t, d = reset(env, rnn_size)
        while not d:
            with torch.no_grad():
                a_t, logp_t, _, v_t, h_t_next = model(o_t, mask_t, h_t)
                # interact with environment
                o, r, d, _ = env.step(a_t.data.item())
                o /= 255.
                total_r += r  # accumulate reward within one rollout.
                # prepare inputs for next step
                mask_t = torch.Tensor([(d + 1) % 2]).cuda()
                o_t = torch.Tensor(o).cuda().unsqueeze(dim=0)  # 128x128 -> 1x128x128
                h_t = h_t_next
        episode_reward.append(total_r)
        print("Episode reward:", total_r)

    print(f"Average eposide reward ({np.mean(episode_reward)})")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default=None)
    args = parser.parse_args()
    if args.model_path is None:
        print("Please specify model path for test")
        sys.exit() 

    env = AI2ThorEnv(config_file="config_files/OneMugTest.json")
    obs_dim = env.observation_space.shape
    # Share information about action space with policy architecture
    rnn_size= 128
    ac_kwargs = dict(hidden_sizes=[64] * 2)
    ac_kwargs['action_space'] = env.action_space
    ac_kwargs['memory_size'] = rnn_size
    # Construct Model
    ac_model = ActorCritic(in_features=obs_dim, **ac_kwargs).cuda()
    state_dict = torch.load(args.model_path)
    # load params
    ac_model.load_state_dict(state_dict)
    tester(env,ac_model,rnn_size)
    env.close()
    print(f"Tester finished job")
