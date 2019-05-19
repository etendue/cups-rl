import sys
import torch
from algorithms.ppo.worker import tester
from algorithms.ppo.core import ActorCritic
from  gym_ai2thor.envs.ai2thor_env import AI2ThorEnv

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
    ac_kwargs = dict()
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
