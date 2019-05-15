from  gym_ai2thor.envs.ai2thor_env import AI2ThorEnv
import numpy as np
import torch
from algorithms.ppo.core import ActorCritic
import sys


def reset(env, state_size):
    o, _, r = env.reset(), False, 0.
    mask_t = torch.tensor(1.,dtype=torch.float32).cuda()
    prev_a = torch.tensor(0, dtype=torch.long).cuda()
    obs_t = torch.Tensor(o/255.).cuda().unsqueeze(dim=0)  # 128x128 -> 1x128x128
    state_t = torch.zeros(state_size,dtype=torch.float32).cuda()
    reward_t = torch.tensor(r,dtype=torch.float32).cuda()
    model_input={"current_obs":obs_t,
                "pre_action":prev_a,
                "pre_state":state_t,
                "state_mask":mask_t}
    return model_input, reward_t

def tester(env, model, rnn_size, n = 5):
    env = AI2ThorEnv(config_file="config_files/OneMugTest.json")
    episode_reward = []
    for _ in range(n):
        # Wait for trainer to inform next job
        total_r = 0.
        d = False
        model_input, _ = reset(env, rnn_size)
        while not d:
            with torch.no_grad():
                a_t, _, _, _, state_t = model(model_input)
                # interact with environment
                o, r, d, _ = env.step(a_t.data.item())
                total_r += r  # accumulate reward within one rollout.
                # prepare inputs for next step
                model_input["state_mask"] = torch.tensor((d+1)%2,dtype=torch.float32).cuda()
                model_input["current_obs"] = torch.Tensor(o/255.).cuda().unsqueeze(dim=0)  # 128x128 -> 1x128x128
                model_input["pre_state"] = state_t
                model_input["pre_action"] = a_t

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
    ac_kwargs['state_size'] = rnn_size
    # Construct Model
    ac_model = ActorCritic(obs_shape=obs_dim, **ac_kwargs).cuda()
    state_dict = torch.load(args.model_path)
    # load params
    ac_model.load_state_dict(state_dict)
    tester(env,ac_model,rnn_size)
    env.close()
    print(f"Tester finished job")
