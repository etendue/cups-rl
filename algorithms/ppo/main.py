import os
import torch
import numpy as np
from torch.multiprocessing import SimpleQueue, Process, Value, Event
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from algorithms.ppo.core import ActorCritic, PPOBuffer, count_vars
from algorithms.ppo.worker import worker
from algorithms.ppo.learner import learner
from gym_ai2thor.envs.ai2thor_env import AI2ThorEnv


def train_ai2thor(model, args, rank=0):

    seed = args.seed + 10000 *rank
    torch.manual_seed(seed)
    np.random.seed(seed)

    # torch.cuda.set_device(rank)
    device = torch.cuda.device(rank)
    os.environ['DISPLAY'] = f':{rank}'

    model = model.to(device)
    model.share_memory()

    # Experience buffer
    storage = PPOBuffer(model.obs_shape, args.steps, args.num_workers, args.state_size, args.gamma, device)
    storage.share_memory()

   
        
    #torch.multiprocessing.set_start_method('spawn')
    # start multiple processes
    ready_to_works = [Event() for _ in range(args.num_workers)]
    exit_flag = Value('i', 0)
    queue = SimpleQueue()

    processes = []
    task_config_file = "config_files/OneMug.json"
    # start workers
    for worker_id in range(args.num_workers):
        p = Process(target=worker, args=(worker_id, model, storage, ready_to_works[worker_id], queue, exit_flag,task_config_file))
        p.start()
        processes.append(p)

    # start trainer
    train_params = {"epochs": args.epochs,
                    "steps": args.steps,
                    "world_size": args.world_size,
                    "num_workers": args.num_workers
                    }
    ppo_params = {"clip_param": args.clip_param,
                  "train_iters": args.train_iters,
                  "mini_batch_size": args.mini_batch_size,
                  "value_loss_coef": args.value_loss_coef,
                  "entropy_coef": args.entropy_coef,
                  "rnn_steps": args.rnn_steps,
                  "lr": args.lr,
                  "max_kl": args.max_kl
                  }
    
    distributed = False
    if args.world_size > 1:
        distributed = True
        # Initialize Process Group, distributed backend type
        dist_backend = 'nccl'
        # Url used to setup distributed training
        dist_url = "tcp://127.0.0.1:23456"
        print("Initialize Process Group... pid:", os.getpid())
        dist.init_process_group(backend=dist_backend, init_method=dist_url, rank=rank, world_size=args.world_size)
        # Make model DistributedDataParallel
        model = DistributedDataParallel(model, device_ids=[rank], output_device=rank)
        from torch.distributed.distributed_c10d import _default_pg
        print(_default_pg)
    learner(model, storage, train_params, ppo_params, ready_to_works, queue, exit_flag, rank, distributed)

    for p in processes:
        print("process ", p.pid, " joined")
        p.join()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--world-size', type=int, default=2)
    parser.add_argument('--steps', type=int, default=1024)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--mini-batch-size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--train-iters', type=int, default=10)
    parser.add_argument('--model-path', type=str, default=None)

    parser.add_argument('--state-size', type=int, default=128)
    parser.add_argument('--rnn-steps', type=int, default=16)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--clip-param', type=float, default=0.2)
    parser.add_argument('--value_loss_coef', type=float, default=0.1)
    parser.add_argument('--entropy_coef', type=float, default=0.001)
    parser.add_argument('--max-kl', type=float, default=0.01)

    args = parser.parse_args()
    torch.multiprocessing.set_start_method('spawn')

    # get observation dimension
    env = AI2ThorEnv(config_file="config_files/OneMug.json")
    obs_dim = env.observation_space.shape
    # Share information about action space with policy architecture
    ac_kwargs = dict()
    ac_kwargs['action_space'] = env.action_space
    ac_kwargs['state_size'] = args.state_size
    env.close()
    # Main model
    print("Initialize Model...")
    # Construct Model
    ac_model = ActorCritic(obs_shape=obs_dim, **ac_kwargs)
    if args.model_path:
        ac_model.load_state_dict(torch.load(args.model_path))
    # Count variables
    var_counts = tuple(count_vars(m) for m in [ac_model.policy, ac_model.value_function, ac_model.feature_base])
    print('\nNumber of parameters: \t pi: %d, \t v: %d \tbase: %d\n' % var_counts)

    if args.world_size > 1:
        processes = []
        for rank in range(args.world_size):
            p = Process(target=train_ai2thor, args=(ac_model, args, rank))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
            print("process ", p.pid, " joined") 
    else:
        train_ai2thor(ac_model, args)
    print("main exits")
