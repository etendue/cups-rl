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


# Proximal Policy Optimization (by clipping),
# with early stopping based on approximate KL

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


def actor(envid, gpuid, model, buf, sync_ev, queue, exitflag):
    print(f"Start actor with pid ({os.getpid()}) ...")

    steps_per_epoch = buf.block_size
    rnn_state_size = buf.h_buf.shape[1]
    torch.cuda.set_device(gpuid)
    env = AI2ThorEnv(config_file="config_files/OneMug.json")
    inputs, r_t = reset(env, rnn_state_size)
    total_r = 0.
    episode_steps = 0
    while True:
        # Wait for next job
        sync_ev.wait()
        if exitflag.value == 1:
            env.close()
            break
        for t in range(steps_per_epoch):
            with torch.no_grad():
                a_t, logp_t, _, v_t, state_t = model(inputs)
                # save experience
                buf.store(envid, inputs["current_obs"], a_t, r_t, v_t, logp_t, inputs["pre_state"], inputs["state_mask"])
                # interact with environment
                o, r, d, _ = env.step(a_t.item())
                total_r += r  # accumulate reward within one rollout.
                episode_steps += 1
                # prepare inputs for next step
                inputs["current_obs"] = torch.Tensor(o/255.).cuda().unsqueeze(dim=0)  # 128x128 -> 1x128x128
                inputs["state_mask"] = torch.tensor((d+1)%2,dtype=torch.float32).cuda()
                inputs["pre_state"] = state_t
                inputs["pre_action"] = a_t
                r_t = torch.tensor(r, dtype=torch.float32).cuda()

                # check terminal state
                epoch_end = (t == (steps_per_epoch - 1))
                if d: # calculate the returns and GAE and reset environment
                    buf.finish_path(envid, r)
                    msg = {"msg_type":"train", "reward":total_r, "steps":episode_steps, "epoch_end":epoch_end}
                    queue.put(msg)
                    inputs, r_t = reset(env, rnn_state_size)
                    total_r, episode_steps = 0., 0
                elif epoch_end: # early cut due to reach maximum steps in on epoch
                    _, _, _, last_val, _ = model(inputs)
                    buf.finish_path(envid, last_val)
                    total_r += last_val.item()
                    msg = {"msg_type":"train", "reward":None, "steps":episode_steps, "epoch_end":epoch_end}
                    queue.put(msg)

        sync_ev.clear()  
    print(f"actor with pid ({os.getpid()})  finished job")

def tester(gpuid, model, rnn_size, sync_ev, queue, exitflag):
    print(f"Start tester with pid ({os.getpid()}) ...")
    torch.cuda.set_device(gpuid)
    env = AI2ThorEnv(config_file="config_files/OneMugTest.json")

    while True:
        # Wait for trainer to inform next job
        total_r, episode_steps, d = 0., 0, False
        inputs, _ = reset(env, rnn_size)
        sync_ev.wait()
        if exitflag.value == 1 :
            env.close()
            break

        while not d:
            with torch.no_grad():
                a_t, _, _, _, state_t = model(inputs)
                # interact with environment
                o, r, d, _ = env.step(a_t.item())
                total_r += r  # accumulate reward within one rollout.
                episode_steps +=1
                # prepare inputs for next step
                inputs["state_mask"] = torch.tensor((d+1)%2, dtype=torch.float32).cuda()
                inputs["current_obs"] = torch.Tensor(o/255.).cuda().unsqueeze(dim=0)  # 128x128 -> 1x128x128
                inputs["pre_state"] = state_t
                inputs["pre_action"] = a_t
                
        msg = {"msg_type": "test", "reward": total_r, "steps": episode_steps, "epoch_end": True}
        queue.put(msg)
        sync_ev.clear()  # Stop working
        
    print(f"Tester with pid ({os.getpid()})  finished job")


def learner(model, buf, params, actor_evs, tester_ev, queue, exitflag):
    print(f"learner with pid ({os.getpid()})  starts job")
    if params.rank == 0:
        writer = SummaryWriter(comment="ai2thor_ppo")

    cr, alpha, beta, target_kl = params.clip_ratio, params.alpha,params.beta ,params.target_kl
    # Optimizers
    optimizer = torch.optim.Adam(model.parameters(), lr=params.lr)
    # average across nodes and multiple gpus

    def dist_sum(x):
        if params.world_size > 1:
            dist.all_reduce(x, dist.ReduceOp.SUM)
        return x

    def dist_mean(x):
        return dist_sum(x)/params.world_size

    # start workers for next epoch
    _ = [ev.set() for ev in actor_evs]
    if params.testing:
        tester_ev.set()
        test_cycle= 5
    # Training policy
    start_time = time.time()
    for epoch in range(params.epochs):
        rollout_ret = []
        rollout_steps = []
        # wait until all workers finish a epoch
        finished_worker = params.num_envs

        if params.testing and epoch%test_cycle == 0:
            test_ret = []
            finished_worker += 1

        while finished_worker > 0:
            msg = queue.get()
            if msg["msg_type"] == "train":
                if msg["reward"] and msg["steps"]:
                    rollout_ret.append(msg["reward"])
                    rollout_steps.append(msg["steps"])
            elif msg["msg_type"] == "test":
                test_ret.append(msg["reward"])

            if msg["epoch_end"]:
                finished_worker -= 1

        # normalize advantage
        # if args.world_size > 1:
        #     mean = exp_buf.adv_buf.mean()
        #     var = exp_buf.adv_buf.var()
        #     mean = dist_mean(mean)
        #     var = dist_mean(var)
        #     exp_buf.normalize_adv(mean_std=(mean, torch.sqrt(var)))
        # else:
        #     exp_buf.normalize_adv()

        # train with batch
        model.train()
        for i in range(params.train_iters):
            batch_gen = buf.batch_generator(params.batch_size)
            kl_sum, ent_sum, pi_loss_sum, v_loss_sum = [torch.tensor(0.0).cuda() for _ in range(4)]

            for batch in batch_gen:
                obs, act, adv, ret, logp_old, pre_state, mask,  pre_a = batch
                model_input = {}
                model_input["state_mask"] = mask
                model_input["current_obs"] = obs
                model_input["pre_state"] = pre_state
                model_input["pre_action"] = pre_a
            
                _, logp_a, ent, v, _ = model(model_input, a=act, horizon_t=params.rnn_steps)
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

            kl_mean = kl_sum / buf.max_size
            kl_mean = dist_mean(kl_mean)
            if torch.abs(kl_mean) > 1.5 * target_kl:
                print(f'Early stopping at iter ({i} /{args.train_iters}) due to reaching max kl. ({kl_mean:.4f})')
                break

        # start workers for next epoch
        model.eval()

        # set the tester, actor processes to exit
        if epoch == params.epochs -1:
            exitflag.value = 1
            if params.testing:
                tester_ev.set()

        if params.testing and (epoch+1)%test_cycle == 0:
            tester_ev.set()
        _ = [ev.set() for ev in actor_evs]

        # calculate statistics
        # kl_mean = kl_sum / exp_buf.max_size
        ent_avg = ent_sum/buf.max_size
        pi_loss_avg = pi_loss_sum/buf.max_size
        v_loss_avg = v_loss_sum/buf.max_size
        # kl_mean = distributed_avg(kl_mean)
        ent_avg = dist_mean(ent_avg)
        pi_loss_avg = dist_mean(pi_loss_avg)
        v_loss_avg = dist_mean(v_loss_avg)
        # Log info about epoch
        global_steps = (epoch + 1)* params.steps * params.world_size
        avg_ret_by_1000 = None
        avg_steps = None
        avg_test_rewards = None

        ret_sum = np.sum(rollout_ret)
        steps_sum = np.sum(rollout_steps)
        count = len(rollout_ret)
        ret_sum = dist_sum(torch.Tensor([ret_sum]).cuda())
        steps_sum = dist_sum(torch.Tensor([steps_sum]).cuda())
        count = dist_sum(torch.Tensor([count]).cuda())

        if count.item() > 0:
            avg_ret_by_1000 = (ret_sum/steps_sum).item() * 1000
            avg_steps = (steps_sum/count).item()
            print(f"Epoch [{epoch}] Steps {global_steps}: "
                  f"return:({avg_ret_by_1000:.1f}), avg_steps:({avg_steps:.1f})")
        else:
            print(f"Epoch [{epoch}] Steps {global_steps}: "
                  f"No reward is collected, very bad")
      
        if params.testing:
            avg_test_rewards = np.mean(test_ret)
            avg_test_rewards = dist_mean(torch.Tensor([avg_test_rewards]).cuda())
            print(f"Epoch [{epoch}] Steps {global_steps}: Rewards1000:({avg_test_rewards.item():.1f})")

        if params.rank == 0:
            fps = global_steps*params.world_size/(time.time()-start_time)
            print(f"Epoch [{epoch}] avg. FPS:[{fps:.2f}]")
            writer.add_scalar("KL", kl_mean, global_steps)
            writer.add_scalar("Entropy", ent_avg, global_steps)
            writer.add_scalar("p_loss", pi_loss_avg, global_steps)
            writer.add_scalar("v_loss", v_loss_avg, global_steps)
            if avg_ret_by_1000:
                writer.add_scalar("Return1000", avg_ret_by_1000, global_steps)
                writer.add_scalar("EpisodeSteps", avg_steps, global_steps)
            if params.testing and avg_test_rewards:
                writer.add_scalar("Rewards1000", avg_test_rewards, global_steps)
            
            if (epoch +1) % 20 == 0:
                if params.world_size > 0:
                    torch.save(model.module.state_dict(), f'model{epoch}.pt') 
                else:
                    torch.save(model.state_dict(), f'model{epoch}.pt') 

    print(f"learner with pid ({os.getpid()})  finished job")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--state-size', type=int, default=128)
    parser.add_argument('--rnn-steps', type=int, default=1)
    parser.add_argument('--world-size', type=int, default=1)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--gpuid', type=int, default=0)
    parser.add_argument('--steps', type=int, default=512)
    parser.add_argument('--num-envs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--model-path', type=str, default=None)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--clip-ratio', type=float, default=0.2)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--beta', type=float, default=0.001)
    parser.add_argument('--train-iters', type=int, default=10)
    parser.add_argument('--testing', action='store_true', default=False)
    parser.add_argument('--target-kl', type=float, default=0.01)

    args = parser.parse_args()

    seed = args.seed
    seed += 10000 * args.rank
    torch.manual_seed(seed)
    np.random.seed(seed)

    torch.multiprocessing.set_start_method('spawn')
    if args.world_size > 1:
        # Initialize Process Group, distributed backend type
        dist_backend = 'nccl'
        # Url used to setup distributed training
        dist_url = "tcp://127.0.0.1:23456"
        print("Initialize Process Group...")
        dist.init_process_group(backend=dist_backend, init_method=dist_url, rank=args.rank, world_size=args.world_size)

    # Establish Local Rank and set device on this node, i.e. the GPU index
    torch.cuda.set_device(args.gpuid)
    # get observation dimension
    env1 = AI2ThorEnv(config_file="config_files/OneMug.json")
    obs_dim = env1.observation_space.shape
    # Share information about action space with policy architecture
    ac_kwargs = dict(hidden_sizes=[args.hid] * args.l)
    ac_kwargs['action_space'] = env1.action_space
    ac_kwargs['state_size'] = args.state_size
    env1.close()
    # Main model
    print("Initialize Model...")
    # Construct Model
    ac_model = ActorCritic(obs_shape=obs_dim, **ac_kwargs).cuda()
    if args.model_path:
        ac_model.load_state_dict(torch.load(args.model_path))
    # share model to multiple processes
    ac_model.share_memory()
    # Count variables
    if args.rank == 0:
        var_counts = tuple(count_vars(m) for m in [ac_model.policy, ac_model.value_function, ac_model.feature_base])
        print('\nNumber of parameters: \t pi: %d, \t v: %d \tbase: %d\n' % var_counts)
    # Experience buffer
    exp_buf = PPOBuffer(obs_dim, args.steps, args.num_envs, args.state_size, args.gamma)
    exp_buf.share_memory()
    # Make model DistributedDataParallel
    if args.world_size > 1:
        d_model = DistributedDataParallel(ac_model, device_ids=[args.gpuid], output_device=args.gpuid)
    else:
        d_model = ac_model
    # start multiple processes
    sync_actor_evs = [Event() for _ in range(args.num_envs)]
    _ = [ev.clear() for ev in sync_actor_evs]
    exit_flag = Value('i',0)
    ret_queue = SimpleQueue()

    processes = []
    #start actors
    for env_id in range(args.num_envs):
        p = Process(target=actor, args=(env_id, args.gpuid, ac_model, exp_buf,
                                        sync_actor_evs[env_id], ret_queue, exit_flag))
        p.start()
        processes.append(p)
    #start tester
    sync_tester_ev = None
    if args.testing:
        sync_tester_ev = Event()
        sync_tester_ev.clear()
        p = Process(target=tester,args=(args.gpuid, ac_model, args.rnn_size, sync_tester_ev, ret_queue, exit_flag))
        p.start()
        processes.append(p)
    # start trainer
    learner(d_model, exp_buf,args, sync_actor_evs, sync_tester_ev, ret_queue, exit_flag)

    for p in processes:
        print("process ", p.pid, " joined")
        p.join()

    print("Main process exits successfully")
