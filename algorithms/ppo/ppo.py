import numpy as np
import torch
import torch.nn.functional as F
import time, os
import scipy.signal
from algorithms.ppo.core import ActorCritic, count_vars

from  gym_ai2thor.envs.ai2thor_env import AI2ThorEnv
from tensorboardX import SummaryWriter

from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
from torch.multiprocessing import SimpleQueue, Event, Process
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, size, num_envs, memory_size, gamma=0.99, lam=0.95):
        self.obs_buf = torch.zeros((size, *obs_dim), dtype=torch.float32).cuda()
        self.act_buf = torch.zeros(size, dtype=torch.float32).cuda()
        self.adv_buf = torch.zeros(size, dtype=torch.float32).cuda()
        self.rew_buf = torch.zeros(size, dtype=torch.float32).cuda()
        self.ret_buf = torch.zeros(size, dtype=torch.float32).cuda()
        self.val_buf = torch.zeros(size, dtype=torch.float32).cuda()
        self.logp_buf = torch.zeros(size, dtype=torch.float32).cuda()
        self.h_buf = torch.zeros((size, memory_size), dtype=torch.float32).cuda()
        self.mask_buf = torch.zeros(size, dtype=torch.float32).cuda()

        # to control the indexing
        self.ptr = torch.zeros(num_envs,dtype=torch.int).cuda()
        self.path_start_idx = torch.zeros(num_envs,dtype=torch.int).cuda()

        # constants
        self.gamma, self.lam, self.max_size, self.block_size = gamma, lam, size, size//num_envs

    def share_memory(self):
        self.obs_buf.share_memory_()
        self.act_buf.share_memory_()
        self.adv_buf.share_memory_()
        self.rew_buf.share_memory_()
        self.ret_buf.share_memory_()
        self.val_buf.share_memory_()
        self.logp_buf.share_memory_()
        self.h_buf.share_memory_()
        self.mask_buf.share_memory_()
        self.ptr.share_memory_()
        self.path_start_idx.share_memory_()

    def store(self, envid, obs, act, rew, val, logp, h, mask):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr[envid].item()  < self.block_size  # buffer has to have room so you can store
        ptr = self.ptr[envid].item()+ envid * self.block_size
        self.obs_buf[ptr].copy_(obs)
        self.act_buf[ptr].copy_(act)
        self.rew_buf[ptr].copy_(rew)
        self.val_buf[ptr].copy_(val)
        self.logp_buf[ptr].copy_(logp)
        self.h_buf[ptr].copy_(h)
        self.mask_buf[ptr].copy_(mask)
        self.ptr[envid] += 1

    def finish_path(self, envid, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """
        # map the index from environment block to whole buffer
        path_start_idx = self.path_start_idx[envid].item() + envid * self.block_size
        ptr = self.ptr[envid].item() + envid * self.block_size
        path_slice = slice(path_start_idx, ptr)

        last_v = torch.Tensor([last_val]).cuda()
        rews = torch.cat((self.rew_buf[path_slice], last_v), dim=0)
        vals = torch.cat((self.val_buf[path_slice], last_v), dim=0)
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = self._discount_cumsum(deltas, self.gamma * self.lam)
        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = self._discount_cumsum(rews, self.gamma)[:-1]
        self.path_start_idx[envid] = self.ptr[envid]

    def normalize_adv(self, mean_std=None, epsilon= 0.0001):
        """
        normalize the advantage with mean and standard deviation. If mean_std is not given, it calculate from date
        :param mean_std:
        :return: None
        """
        if mean_std == None:
            mean = self.adv_buf.mean()
            std = self.adv_buf.std()
        else:
            mean= mean_std[0]
            std = mean_std[1]
        self.adv_buf = (self.adv_buf - mean)/(std +epsilon)

    def get_batch(self, batch_size, rnn_steps):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer. Also, resets some pointers in the buffer.
        """
        if self.ptr.sum().item() != 0:
            assert self.ptr.sum().item() == self.max_size  # buffer has to be full before you can get
            self.ptr.copy_(torch.zeros_like(self.ptr))
            self.path_start_idx.copy_(torch.zeros_like(self.path_start_idx))

        num_batch = self.max_size//batch_size
        num_clusters = self.max_size // rnn_steps
        indice = np.random.permutation(num_clusters).reshape(num_batch, -1)
        for idx in indice:
            sl = np.hstack([np.arange(i, i + rnn_steps) for i in idx])
            yield [
                self.obs_buf[sl], self.act_buf[sl], self.adv_buf[sl], self.ret_buf[sl],
                self.logp_buf[sl], self.h_buf[sl], self.mask_buf[sl]
            ]

    def batch_generator(self, batch_size):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer. Also, resets some pointers in the buffer.
        """
        if self.ptr.sum().item() != 0:
            assert self.ptr.sum().item() == self.max_size  # buffer has to be full before you can get
            self.ptr.copy_(torch.zeros_like(self.ptr))
            self.path_start_idx.copy_(torch.zeros_like(self.path_start_idx))

        batch_sampler = BatchSampler( SubsetRandomSampler(range(self.max_size)), batch_size, drop_last=False)
        for idx in batch_sampler:
            yield [
                self.obs_buf[idx], self.act_buf[idx], self.adv_buf[idx], self.ret_buf[idx],
                self.logp_buf[idx], self.h_buf[idx], self.mask_buf[idx]
            ]

    def _discount_cumsum(self, x, discount):
        """
        magic from rllab for computing discounted cumulative sums of vectors.
        input: vector x,[x0,x1,x2]
        output: [x0 + discount * x1 + discount^2 * x2, x1 + discount * x2, x2]
        """
        flipped_x = torch.flip(x,dims=(0,)).cpu()
        out = scipy.signal.lfilter([1], [1, float(-discount)], flipped_x, axis=0)
        t = torch.from_numpy(out).cuda()
        return torch.flip(t, dims=(0,))

"""
Proximal Policy Optimization (by clipping),

with early stopping based on approximate KL

"""


def actor(env_id, gpuid, model, exp_buf, epochs, sync_ev, ret_queue):
    print(f"Start actor with pid ({os.getpid()}) ...")

    steps_per_epoch = exp_buf.block_size
    rnn_state_size = exp_buf.h_buf.shape[1]
    torch.cuda.set_device(gpuid)

    env = AI2ThorEnv(config_file="config_files/OneMug.json")
    o, d, r = env.reset() / 255., False, 0.
    mask_t = torch.Tensor([1.]).cuda()
    o_t = torch.Tensor(o).cuda().unsqueeze(dim=0) # 128x128 -> 1x128x128
    h_t = torch.zeros(rnn_state_size).cuda()
    r_t = torch.Tensor([r]).cuda()
    total_r = 0.
    episode_steps = 0

    for _ in range(epochs):
        # Wait for trainer to inform next job
        sync_ev.wait()
        with torch.no_grad():
            for t in range(steps_per_epoch):
                a_t, logp_t, _, v_t, h_t_next = model(o_t, mask_t, h_t)
                # save experience
                exp_buf.store(env_id, o_t, a_t, r_t[0], v_t, logp_t, h_t, mask_t[0])
                # interact with environment
                o, r, d, _ = env.step(a_t.data.item())
                o /= 255.
                total_r += r  # accumulate reward within one rollout.
                episode_steps +=1
                # prepare inputs for next step
                mask_t = torch.Tensor([(d+1)%2]).cuda()
                o_t = torch.Tensor(o).cuda().unsqueeze(dim=0)  # 128x128 -> 1x128x128
                h_t = h_t_next
                r_t = torch.Tensor([r]).cuda()
                # check terminal state
                epoch_end = (t == (steps_per_epoch - 1))
                if d: # calculate the returns and GAE and reset environment
                    exp_buf.finish_path(env_id, 0.)
                    o, d, r = env.reset() / 255., False, 0.
                    mask_t = torch.Tensor([1.]).cuda()
                    o_t = torch.Tensor(o).cuda().unsqueeze(dim=0)  # 128x128 -> 1x128x128
                    h_t = torch.zeros(rnn_state_size).cuda()
                    r_t = torch.Tensor([r]).cuda()
                    ret_queue.put((total_r, episode_steps, epoch_end))
                    total_r = 0.
                    episode_steps = 0
                elif epoch_end: # early cut due to reach maximum steps in on epoch
                    _, _, _, last_val, _ = model(o_t, mask_t, h_t)
                    exp_buf.finish_path(env_id, last_val)
                    ret_queue.put((None, None, epoch_end))

        sync_ev.clear()  # Stop working

    sync_ev.wait() # wait to exits.
    env.close()
    print(f"actor with pid ({os.getpid()})  finished job")


def learner(model, exp_buf, sync_evs, ret_queue, args):
    print(f"learner with pid ({os.getpid()})  starts job")
    if args.rank == 0:
        writer = SummaryWriter(comment="ai2thor_ppo")

    cr, alpha, beta, target_kl = args.clip_ratio,args.alpha,args.beta,0.01
    # Optimizers
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # average across nodes and multiple gpus

    def distributed_avg(x):
        if args.world_size > 1:
            if not isinstance(x,torch.Tensor) or not x.is_cuda:
                x_t = torch.Tensor(x).cuda()
                dist.all_reduce(x_t,dist.ReduceOp.SUM)
                x = x_t.item()
            else:
                dist.all_reduce(x, dist.ReduceOp.SUM)
            x /= dist.get_world_size()
        return x

    # start workers for next epoch
    [ev.set() for ev in sync_evs]
    # Training policy
    start_time = time.time()
    for epoch in range(args.epochs):
        finished_worker = 0
        rollout_ret = []
        rollout_steps = []
        # wait until all workers finish a epoch
        while finished_worker < args.num_envs :
            ret, steps, epoch_end = ret_queue.get()
            if epoch_end:
                finished_worker += 1
            if ret is not None:
                rollout_ret.append(ret)
                rollout_steps.append(steps)

        # normalize advantage
        if args.world_size > 1:
            mean = exp_buf.adv_buf.mean()
            var = exp_buf.adv_buf.var()
            dist.all_reduce(mean, dist.ReduceOp.SUM)
            dist.all_reduce(var,dist.ReduceOp.SUM)
            mean /= args.world_size
            var /= args.world_size
            exp_buf.normalize_adv(mean_std=(mean, torch.sqrt(var)))
        else:
            exp_buf.normalize_adv()

        # train with batch
        model.train()
        for i in range(args.train_iters):
            batch_gen = exp_buf.batch_generator(args.batch_size)
            kl_sum, ent_sum, pi_loss_sum,v_loss_sum= .0, .0, .0, .0

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
                entropy = ent.mean()
                # Policy gradient step
                optimizer.zero_grad()
                (pi_loss + v_loss * alpha - entropy * beta).backward()
                optimizer.step()
                with torch.no_grad():
                    batch_size = len(batch)
                    kl = (logp_old - logp_a).sum()
                    if torch.isnan(kl):
                        print("SOMETHING VERY WRONG HERE")
                    kl_sum += kl.item()
                    ent_sum += entropy.item() * batch_size
                    pi_loss_sum += pi_loss.item() * batch_size
                    v_loss_sum += v_loss.item() * batch_size

            kl_mean = kl_sum / exp_buf.max_size
            kl_mean = distributed_avg(kl_mean)
            print(f'Mean KL:{kl_mean:.4f}')
            if kl_mean > 1.5 * target_kl:
                #print(f'Early stopping at iter ({i} /{args.train_iters}) due to reaching max kl. ({kl_mean:.4f})')
                #break
                print(f'KL divergence exeeds target KL value {kl_mean:.4f} > 1.5 x {target_kl:.4f}')
                break

        # start workers for next epoch
        model.eval()
        [ev.set() for ev in sync_evs]

        # calculate statistics
        ent_avg = ent_sum/exp_buf.max_size
        pi_loss_avg = pi_loss_sum/exp_buf.max_size
        v_loss_avg = v_loss_sum/exp_buf.max_size
        ent_avg = distributed_avg(ent_avg)
        pi_loss_avg = distributed_avg(pi_loss_avg)
        v_loss_avg = distributed_avg(v_loss_avg)
        # Log info about epoch
        global_steps = (epoch + 1)* args.steps * args.world_size
        if args.rank == 0:
            fps = global_steps*args.world_size/(time.time()-start_time)
            print(f"Epoch [{epoch}] avg. FPS:[{fps:.2f}]")
            writer.add_scalar("KL", kl_mean, global_steps)
            writer.add_scalar("Entropy", ent_avg, global_steps)
            writer.add_scalar("p_loss", pi_loss_avg, global_steps)
            writer.add_scalar("v_loss", v_loss_avg, global_steps)
        if len(rollout_ret) >0:
            ret_by_1000 = np.sum(rollout_ret)*1000/exp_buf.max_size
            print(f"Epoch [{epoch}] Steps {global_steps}: return:({ret_by_1000:.1f}), avg_steps:({np.mean(rollout_steps):.1f})")
        else:
            print(f"Epoch [{epoch}] Steps {global_steps}: does not have finished rollouts")

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
    model = ActorCritic(in_features=obs_dim, **ac_kwargs).cuda()
    # share model to multiple processes
    model.share_memory()
    # Count variables
    if args.rank == 0:
        var_counts = tuple( count_vars(module)
                            for module in [model.policy, model.value_function, model.feature_base])
        print('\nNumber of parameters: \t pi: %d, \t v: %d \tbase: %d\n' % var_counts)
    # Experience buffer
    buf = PPOBuffer(obs_dim, args.steps, args.num_envs, args.rnn_size, args.gamma)
    buf.share_memory()
    # Make model DistributedDataParallel
    if args.world_size > 1:
        d_model = DistributedDataParallel(model, device_ids=[args.gpuid], output_device=args.gpuid)
    else:
        d_model = model
    # start multiple processes
    sync_evs = [Event() for _ in range(args.num_envs)]
    [ev.clear() for ev in sync_evs]
    ret_queue = SimpleQueue()

    processes = []
    for env_id in range(args.num_envs):
        p = Process(target=actor, args=(env_id, args.gpuid, model, buf, args.epochs, sync_evs[env_id], ret_queue))
        p.start()
        processes.append(p)
    # start trainer
    learner(d_model, buf, sync_evs, ret_queue, args)

    for p in processes:
        print("process ", p.pid, " joined")
        p.join()

    print("Main process exits successfully")
