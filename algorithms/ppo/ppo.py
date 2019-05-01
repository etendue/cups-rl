import numpy as np
import torch
import torch.nn.functional as F
import time, os
from algorithms.ppo.core import ActorCritic, PPOBuffer, count_vars

from  gym_ai2thor.envs.ai2thor_env import AI2ThorEnv
from tensorboardX import SummaryWriter

from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
from torch.multiprocessing import SimpleQueue, Event, Process, Value

"""
Proximal Policy Optimization (by clipping),

with early stopping based on approximate KL

"""


def reset(env, rnn_size):
    o, d, r = env.reset() / 255., False, 0.
    mask_t = torch.Tensor([1.]).cuda()
    o_t = torch.Tensor(o).cuda().unsqueeze(dim=0)  # 128x128 -> 1x128x128
    h_t = torch.zeros(rnn_size).cuda()
    r_t = torch.Tensor([r]).cuda()
    return o_t, r_t, h_t, mask_t, d


def actor(env_id, gpuid, model, buf, epochs, sync_ev, queue, exit_flag):
    print(f"Start actor with pid ({os.getpid()}) ...")

    steps_per_epoch = buf.block_size
    rnn_state_size = buf.h_buf.shape[1]
    torch.cuda.set_device(gpuid)

    env = AI2ThorEnv(config_file="config_files/OneMug.json")
    o_t, r_t, h_t, mask_t, d = reset(env, rnn_state_size)
    total_r = 0.
    episode_steps = 0

    while True:
        # Wait for trainer to inform next job
        sync_ev.wait()
        if exit_flag.value == 1:
            env.close()
            break

        with torch.no_grad():
            for t in range(steps_per_epoch):
                a_t, logp_t, _, v_t, h_t_next = model(o_t, mask_t, h_t)
                # save experience
                buf.store(env_id, o_t, a_t, r_t[0], v_t, logp_t, h_t, mask_t[0])
                # interact with environment
                o, r, d, _ = env.step(a_t.data.item())
                o /= 255.
                total_r += r  # accumulate reward within one rollout.
                episode_steps += 1
                # prepare inputs for next step
                mask_t = torch.Tensor([(d+1)%2]).cuda()
                o_t = torch.Tensor(o).cuda().unsqueeze(dim=0)  # 128x128 -> 1x128x128
                h_t = h_t_next
                r_t = torch.Tensor([r]).cuda()
                # check terminal state
                epoch_end = (t == (steps_per_epoch - 1))
                if d: # calculate the returns and GAE and reset environment
                    buf.finish_path(env_id, 0.)
                    o_t, r_t, h_t, mask_t, d = reset(env, rnn_state_size)
                    msg = {"msg_type":"train", "reward":total_r, "steps":episode_steps, "epoch_end":epoch_end}
                    queue.put(msg)
                    total_r = 0.
                    episode_steps = 0
                elif epoch_end: # early cut due to reach maximum steps in on epoch
                    _, _, _, last_val, _ = model(o_t, mask_t, h_t)
                    buf.finish_path(env_id, last_val)
                    msg = {"msg_type":"train","reward":None, "steps":None, "epoch_end":epoch_end}
                    queue.put(msg)
        sync_ev.clear()  # Stop working

    print(f"actor with pid ({os.getpid()})  finished job")


def tester(gpuid, model, rnn_size, sync_ev, queue, exit_flag):
    print(f"Start tester with pid ({os.getpid()}) ...")
    torch.cuda.set_device(gpuid)
    env = AI2ThorEnv(config_file="config_files/OneMugTest.json")

    while True:
        # Wait for trainer to inform next job
        episode_steps = 0
        total_r = 0.
        o_t, r_t, h_t, mask_t, d = reset(env, rnn_size)

        sync_ev.wait()
        print(exit_flag.value)
        if exit_flag.value == 1 :
            env.close()
            break

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
                episode_steps +=1

        msg = {"msg_type": "test", "reward": total_r, "steps": episode_steps, "epoch_end": True}
        queue.put(msg)
        sync_ev.clear()  # Stop working

    print(f"Tester with pid ({os.getpid()})  finished job")


def learner(model, exp_buf, args, sync_evs, sync_tester_ev,ret_queue, exit_flag):
    print(f"learner with pid ({os.getpid()})  starts job")
    if args.rank == 0:
        writer = SummaryWriter(comment="ai2thor_ppo")

    cr, alpha, beta, target_kl = args.clip_ratio,args.alpha,args.beta,0.01
    # Optimizers
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # average across nodes and multiple gpus

    def distributed_avg(x):
        if args.world_size > 1:
            dist.all_reduce(x, dist.ReduceOp.SUM)
            x /= dist.get_world_size()
        return x

    # start workers for next epoch
    [ev.set() for ev in sync_evs]
    sync_tester_ev.set()
    test_cycle= 5
    # Training policy
    start_time = time.time()
    for epoch in range(args.epochs):
        rollout_ret = []
        rollout_steps = []
        test_ret = []
        # wait until all workers finish a epoch
        finished_worker = args.num_envs

        if epoch%test_cycle == 0:
            finished_worker += 1

        while finished_worker > 0:
            msg = ret_queue.get()
            if msg["msg_type"] == "train":
                if msg["reward"] and msg["steps"]:
                    rollout_ret.append(msg["reward"])
                    rollout_steps.append(msg["steps"])
            elif msg["msg_type"] == "test":
                test_ret.append(msg["reward"])

            if msg["epoch_end"]:
                finished_worker -= 1

        # normalize advantage
        if args.world_size > 1:
            mean = exp_buf.adv_buf.mean()
            var = exp_buf.adv_buf.var()
            dist.all_reduce(mean, dist.ReduceOp.SUM)
            dist.all_reduce(var, dist.ReduceOp.SUM)
            mean /= args.world_size
            var /= args.world_size
            exp_buf.normalize_adv(mean_std=(mean, torch.sqrt(var)))
        else:
            exp_buf.normalize_adv()

        # train with batch
        model.train()
        for i in range(args.train_iters):
            batch_gen = exp_buf.batch_generator(args.batch_size)
            kl_sum, ent_sum, pi_loss_sum, v_loss_sum = [torch.tensor(0.0).cuda() for _ in range(4)]

            for batch in batch_gen:
                obs, act, adv, ret, logp_old, h, mask = batch
                _, logp_a, ent, v, _ = model(obs, mask, h, a=act, horizon_t=args.rnn_steps)
                # PPO policy objective
                ratio = (logp_a - logp_old).exp()
                min_adv = torch.where(adv > 0, (1 + cr) * adv, (1 - cr) * adv)
                pi_loss = -(torch.min(ratio * adv, min_adv)).mean()
                # PPO value objective
                v_loss = F.mse_loss(v, ret)
                # PPO entropy objective
                ent_mean = ent.mean()
                # Policy gradient step
                optimizer.zero_grad()
                (pi_loss + v_loss * alpha - ent_mean * beta).backward()
                optimizer.step()
                with torch.no_grad():
                    batch_size = len(act)
                    kl_sum += (logp_old - logp_a).sum()
                    ent_sum += ent_mean * batch_size
                    pi_loss_sum += pi_loss * batch_size
                    v_loss_sum += v_loss * batch_size

            # kl_mean = kl_sum / exp_buf.max_size
            # kl_mean = distributed_avg(kl_mean)
            # if kl_mean > 1.5 * target_kl:
                #print(f'Early stopping at iter ({i} /{args.train_iters}) due to reaching max kl. ({kl_mean:.4f})')
                #break

        # start workers for next epoch
        model.eval()

        # set the tester, actor processes to exit
        if epoch == args.epochs -1:
            exit_flag.value = 1
            sync_tester_ev.set()

        if (epoch+1)%test_cycle == 0:
            sync_tester_ev.set()
        [ev.set() for ev in sync_evs]

        # calculate statistics
        kl_mean = kl_sum / exp_buf.max_size
        ent_avg = ent_sum/exp_buf.max_size
        pi_loss_avg = pi_loss_sum/exp_buf.max_size
        v_loss_avg = v_loss_sum/exp_buf.max_size
        kl_mean = distributed_avg(kl_mean)
        ent_avg = distributed_avg(ent_avg)
        pi_loss_avg = distributed_avg(pi_loss_avg)
        v_loss_avg = distributed_avg(v_loss_avg)
        # Log info about epoch
        global_steps = (epoch + 1)* args.steps * args.world_size
        ret_by_1000 = None
        avg_steps = None
        avg_test_rewards = None

        if kl_mean > 1.5 * target_kl:
            print(f'KL divergence exeeds target KL value {kl_mean:.4f} > 1.5 x {target_kl:.4f}')
        else:
            print(f'KL divergence :{kl_mean:.4f}')

        if len(rollout_ret) > 0:
            ret_by_1000 = np.sum(rollout_ret)*1000/exp_buf.max_size
            avg_steps = np.mean(rollout_steps)
            ret_by_1000 = distributed_avg(torch.Tensor([ret_by_1000]).cuda())
            avg_steps  = distributed_avg(torch.Tensor([avg_steps]).cuda())
            print(f"Epoch [{epoch}] Steps {global_steps}: "
                  f"return:({ret_by_1000.item():.1f}), avg_steps:({avg_steps.item():.1f})")
        else:
            print(f"Epoch [{epoch}] Steps {global_steps}: does not have finished rollouts")

        if test_ret:
            avg_test_rewards = np.mean(test_ret)
            avg_test_rewards = distributed_avg(torch.Tensor([avg_test_rewards]).cuda())
            print(f"Epoch [{epoch}] Steps {global_steps}: Rewards1000:({avg_test_rewards.item():.1f})")

        if args.rank == 0:
            fps = global_steps*args.world_size/(time.time()-start_time)
            print(f"Epoch [{epoch}] avg. FPS:[{fps:.2f}]")
            writer.add_scalar("KL", kl_mean, global_steps)
            writer.add_scalar("Entropy", ent_avg, global_steps)
            writer.add_scalar("p_loss", pi_loss_avg, global_steps)
            writer.add_scalar("v_loss", v_loss_avg, global_steps)
            if ret_by_1000:
                writer.add_scalar("Return1000", ret_by_1000, global_steps)
                writer.add_scalar("EpisodeSteps", avg_steps, global_steps)
            if avg_test_rewards:
                writer.add_scalar("Rewards1000", avg_test_rewards, global_steps)

    print(f"learner with pid ({os.getpid()})  finished job")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--rnn-size', type=int, default=128)
    parser.add_argument('--rnn-steps', type=int, default=1)
    parser.add_argument('--world-size', type=int, default=1)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--gpuid', type=int, default=0)
    parser.add_argument('--steps', type=int, default=2048)
    parser.add_argument('--num-envs', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=100)
    # parser.add_argument('--model-path', type=str, default=None)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--clip-ratio', type=float, default=0.2)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--beta', type=float, default=0.01)
    parser.add_argument('--train-iters', type=int, default=1)

    args = parser.parse_args()

    seed = args.seed
    seed += 10000 * args.rank
    torch.manual_seed(seed)
    np.random.seed(seed)

    torch.multiprocessing.set_start_method('spawn')
    if args.world_size > 1:
        # Initialize Process Group
        # Distributed backend type
        dist_backend = 'nccl'
        # Url used to setup distributed training
        dist_url = "tcp://127.0.0.1:23456"
        print("Initialize Process Group...")
        dist.init_process_group(backend=dist_backend, init_method=dist_url, rank=args.rank, world_size=args.world_size)

    # Establish Local Rank and set device on this node, i.e. the GPU index
    torch.cuda.set_device(args.gpuid)
    # get observation dimension
    env = AI2ThorEnv(config_file="config_files/OneMug.json")
    obs_dim = env.observation_space.shape
    # Share information about action space with policy architecture
    ac_kwargs = dict(hidden_sizes=[args.hid] * args.l)
    ac_kwargs['action_space'] = env.action_space
    ac_kwargs['memory_size'] = args.rnn_size
    env.close()
    # Main model
    print("Initialize Model...")
    # Construct Model
    ac_model = ActorCritic(in_features=obs_dim, **ac_kwargs).cuda()
    # share model to multiple processes
    ac_model.share_memory()
    # Count variables
    if args.rank == 0:
        var_counts = tuple( count_vars(module)
                            for module in [ac_model.policy, ac_model.value_function, ac_model.feature_base])
        print('\nNumber of parameters: \t pi: %d, \t v: %d \tbase: %d\n' % var_counts)
    # Experience buffer
    buf = PPOBuffer(obs_dim, args.steps, args.num_envs, args.rnn_size, args.gamma)
    buf.share_memory()
    # Make model DistributedDataParallel
    if args.world_size > 1:
        d_model = DistributedDataParallel(ac_model, device_ids=[args.gpuid], output_device=args.gpuid)
    else:
        d_model = ac_model
    # start multiple processes
    sync_actor_evs = [Event() for _ in range(args.num_envs)]
    [ev.clear() for ev in sync_actor_evs]
    sync_tester_ev = Event()
    sync_tester_ev.clear()

    exit_flag = Value('i',0)
    ret_queue = SimpleQueue()

    processes = []
    #start actors
    for env_id in range(args.num_envs):
        p = Process(target=actor, args=(env_id, args.gpuid, ac_model, buf, args.epochs,
                                        sync_actor_evs[env_id], ret_queue,exit_flag))
        p.start()
        processes.append(p)

    #start tester
    p = Process(target=tester,args=(args.gpuid, ac_model, args.rnn_size, sync_tester_ev, ret_queue, exit_flag))
    p.start()
    processes.append(p)
    # start trainer
    learner(d_model, buf,args, sync_actor_evs, sync_tester_ev, ret_queue, exit_flag)

    for p in processes:
        print("process ", p.pid, " joined")
        p.join()

    print("Main process exits successfully")
