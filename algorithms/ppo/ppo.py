import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
import time
import scipy.signal
from algorithms.ppo.core import ActorCritic, count_vars

from  gym_ai2thor.envs.ai2thor_env import AI2ThorEnv
from tensorboardX import SummaryWriter
from collections import deque

from  torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist


def log_info(msg):
    if rank == 0:
        print(msg)


def distributed_avg(x):
    if world_size >1:
        dist.all_reduce(x,dist.ReduceOp.SUM)
        return x/world_size
    else:
        return x


class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, size, memory_size, gamma=0.99, lam=0.95):
        self.obs_buf = torch.zeros(self._combined_shape(size,obs_dim),dtype=torch.float32).cuda()
        self.act_buf = torch.zeros(size,dtype=torch.float32).cuda()
        self.adv_buf = torch.zeros(size,dtype=torch.float32).cuda()
        self.rew_buf = torch.zeros(size,dtype=torch.float32).cuda()
        self.ret_buf = torch.zeros(size,dtype=torch.float32).cuda()
        self.val_buf = torch.zeros(size,dtype=torch.float32).cuda()
        self.logp_buf = torch.zeros(size,dtype=torch.float32).cuda()
        self.h_buf = torch.zeros((size, memory_size), dtype=torch.float32).cuda()
        self.mask_buf = torch.zeros(size,dtype=torch.float32).cuda()

        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp, h, mask):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size  # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.h_buf[self.ptr] = h
        self.mask_buf[self.ptr] = mask
        self.ptr += 1

    def finish_path(self, last_val=0):
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

        path_slice = slice(self.path_start_idx, self.ptr)
        last_v = Tensor([last_val]).cuda()
        rews = torch.cat([self.ret_buf[path_slice], last_v], dim=0)
        vals = torch.cat([self.val_buf[path_slice], last_v], dim=0)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = self._discount_cumsum(deltas, self.gamma * self.lam)

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = self._discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size  # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = self._mean_std(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        return [
            self.obs_buf, self.act_buf, self.adv_buf, self.ret_buf,
            self.logp_buf, self.h_buf, self.mask_buf
        ]

    def get_batch(self, batch_size, time_horizon):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        if self.ptr != 0:
            assert self.ptr == self.max_size  # buffer has to be full before you can get
            self.ptr, self.path_start_idx = 0, 0
            # the next two lines implement the advantage normalization trick
            adv_mean, adv_std = self._mean_std(self.adv_buf)
            self.adv_buf = (self.adv_buf - adv_mean) / adv_std

        num_batch = self.max_size//batch_size
        num_block = self.max_size//time_horizon
        indice = np.random.permutation(num_block).reshape(num_batch, -1)

        for idx in indice:
            sl = np.hstack([np.arange(i, i+time_horizon) for i in idx])
            yield [
                self.obs_buf[sl], self.act_buf[sl], self.adv_buf[sl], self.ret_buf[sl],
                self.logp_buf[sl], self.h_buf[sl], self.mask_buf[sl]
            ]

    def _combined_shape(self, length, shape=None):
        if shape is None:
            return (length, )
        return (length, shape) if np.isscalar(shape) else (length, *shape)

    def _discount_cumsum(self, x, discount):
        """
        magic from rllab for computing discounted cumulative sums of vectors.
        input:
            vector x,[x0,x1,x2]

        output:
            [x0 + discount * x1 + discount^2 * x2, x1 + discount * x2, x2]
        """
        flipped_x = torch.flip(x,dims=(0,)).cpu()
        out=  scipy.signal.lfilter([1], [1, float(-discount)], flipped_x, axis=0)
        t= torch.from_numpy(out).cuda()
        return torch.flip(t,dims=(0,))

    def _mean_std(self, x):
        mean = torch.mean(x,dim=-1)
        mean = distributed_avg(mean)

        var = torch.mean((x-mean)**2)
        var = distributed_avg(var)

        return mean, torch.sqrt(var)



"""

Proximal Policy Optimization (by clipping),

with early stopping based on approximate KL

"""


def ppo(env_fn,
        model=ActorCritic,
        ac_kwargs=dict(),
        seed=0,
        steps_per_epoch=4000,
        epochs=50,
        gamma=0.99,
        clip_ratio=0.2,
        lr=3e-4,
        train_iters=40,
        alpha=0.1,
        beta=0.01,
        lam=0.97,
        target_kl=0.01,
        save_freq=10,
        model_path=None):
    """

    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        ac_model: The agent's main model which is composed of
            the policy and value function model, where the policy takes
            some state, ``x`` and action ``a``, and value function takes
            the state ``x``. The model returns a tuple of:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``a``       (batch, act_dim)  | Samples actions from policy given
                                           | states.
            ``logp_a``     (batch,)          | Gives log probability, according to
                                           | the policy, of taking actions ``a``
                                           | in states ``x``.
            ``ent``      (batch,)          | Entropy of probablity, according to
                                           | the policy
                                           |
            ``v``        (batch,)          | Gives the value estimate for states
                                           | in ``x``. (Critical: make sure
                                           | to flatten this via .squeeze()!)
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the actor_critic
            class you provided to PPO.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs)
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs of interaction (equivalent to
            number of policy updates) to perform.

        gamma (float): Discount factor. (Always between 0 and 1.)

        clip_ratio (float): Hyperparameter for clipping in the policy objective.
            Roughly: how far can the new policy go from the old policy while
            still profiting (improving the objective function)? The new policy
            can still go farther than the clip_ratio says, but it doesn't help
            on the objective anymore. (Usually small, 0.1 to 0.3.)

        lr (float): Learning rate for  optimizer.

        train_iters (int): Maximum number of gradient descent steps to take
            on policy loss per epoch. (Early stopping may cause optimizer
            to take fewer than this.)

        alpha (float) : ratio of loss objective for value

        beta (float) : ratio of loss objective for entropy

        lam (float): Lambda for GAE-Lambda. (Always between 0 and 1,
            close to 1.)


        target_kl (float): Roughly what KL divergence we think is appropriate
            between new and old policies after an update. This will get used
            for early stopping. (Usually small, 0.01 or 0.05.)

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

        model_path(str): path to load saved model

    """
    # the memory_size for RNN hidden state
    memory_size = 256
    # time horizon length for RNN
    horizon_t = 20
    batch_size = args.batch_size

    seed += 10000 * rank
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = env_fn()
    obs_dim = env.observation_space.shape

    # Share information about action space with policy architecture
    ac_kwargs['action_space'] = env.action_space

    # Main model
    print("Initialize Model...")
    # Construct Model

    ac_model = model(in_features=obs_dim, **ac_kwargs).cuda()
    #if model_path:
    #    actor_critic.load_state_dict(torch.load(model_path),map_location=device)

    # Experience buffer
    local_steps_per_epoch = int(steps_per_epoch / world_size)
    buf = PPOBuffer(obs_dim, local_steps_per_epoch, memory_size, gamma, lam)

    # Count variables
    var_counts = tuple(
        count_vars(module)
        for module in [ac_model.policy, ac_model.value_function, ac_model.feature_base])
    log_info('\nNumber of parameters: \t pi: %d, \t v: %d \tbase: %d\n' % var_counts)
    
    # Make model DistributedDataParallel
    if world_size >1:
        ac_model = DistributedDataParallel(ac_model, device_ids=[args.local_rank], output_device=args.local_rank)

    # Optimizers
    optimizer = torch.optim.Adam(ac_model.parameters(), lr=lr)


    # add TensorBoard
    if rank == 0:
        writer = SummaryWriter(comment="ai2thor_pp")

    def update_with_batch():

        # Training policy
        for i in range(train_iters):
            batch_gen = buf.get_batch(batch_size,horizon_t)
            kl_batches = []
            ent_batches = []
            pi_loss_batches=[]
            v_loss_batches=[]
            for batch in batch_gen:
                obs, act, adv, ret, logp_old, h, mask = batch
                # Output from policy function graph
                # x, _ = ac_model.process_feature(obs, mask, h, horizon_t)
                #_, logp_a, ent = ac_model.policy(x, act)
                # v = ac_model.value_function(x)
                _, logp_a,ent,v,_ = ac_model(obs,mask,h,a=act,horizon_t=horizon_t)
                # PPO policy objective
                ratio = (logp_a - logp_old).exp()
                min_adv = torch.where(adv > 0, (1 + clip_ratio) * adv, (1 - clip_ratio) * adv)
                pi_loss = -(torch.min(ratio * adv, min_adv)).mean()
                # PPO value objective
                v_loss = F.mse_loss(v, ret)
                # PPO entropy objective
                entropy = ent.mean()

                # Policy gradient step
                optimizer.zero_grad()
                (pi_loss + v_loss * alpha - entropy * beta).backward()
                optimizer.step()

                # KL divergence
                kl = (logp_old - logp_a).mean().detach()

                kl_batches.append(kl)
                ent_batches.append(entropy)
                pi_loss_batches.append(pi_loss)
                v_loss_batches.append(v_loss)

            kl_mean = torch.mean(torch.stack(kl_batches,dim=0))
            kl = distributed_avg(kl_mean)
            if kl > 1.5 * target_kl:
                log_info(f'Early stopping at step ({i}) due to reaching max kl. ({kl.data.item():.4})')
                break
        ent_avg = torch.mean(torch.stack(ent_batches))
        pi_loss_avg = torch.mean(torch.stack(pi_loss_batches))
        v_loss_avg = torch.mean(torch.stack(v_loss_batches))
        ent_avg = distributed_avg(ent_avg)
        pi_loss_avg = distributed_avg(pi_loss_avg)
        v_loss_avg = distributed_avg(v_loss_avg)
        # Log info about epoch TODO
        if rank == 0:
            writer.add_scalar("KL", kl, global_steps)
            writer.add_scalar("Entropy", ent_avg, global_steps)
            writer.add_scalar("p_loss", pi_loss_avg, global_steps)
            writer.add_scalar("v_loss", v_loss_avg, global_steps)

    def update():
        obs, act, adv, ret, logp_old, h, mask = buf.get()

        # Training policy
        for i in range(train_iters):
            # Output from policy function graph
            # x,_ = ac_model.process_feature(obs, mask, h, horizon_t)
            # _, logp_a, ent = ac_model.policy(x, act)
            # v = ac_model.value_function(x)
            _, logp_a,ent,v,_ = ac_model(obs,mask,h,a=act,horizon_t=horizon_t)
            # PPO policy objective
            ratio = (logp_a - logp_old).exp()
            min_adv = torch.where(adv > 0, (1 + clip_ratio) * adv,
                                  (1 - clip_ratio) * adv)
            pi_loss = -(torch.min(ratio * adv, min_adv)).mean()
            # PPO value objective
            v_loss = F.mse_loss(v, ret)
            # PPO entropy objective
            entropy = ent.mean()
            # KL divergence
            kl = (logp_old - logp_a).mean().detach()
            kl = distributed_avg(kl)
            if kl > 1.5 * target_kl:
                log_info(f'Early stopping at step ({i}) due to reaching max kl. ({kl:.4})')
                break

            # Policy gradient step
            optimizer.zero_grad()
            (pi_loss + v_loss * alpha - entropy*beta).backward()
            optimizer.step()

        # Log info about epoch
        if rank == 0:
            writer.add_scalar("KL",kl,global_steps)
            writer.add_scalar("Entropy", entropy, global_steps)
            writer.add_scalar("p_loss", pi_loss, global_steps)
            writer.add_scalar("v_loss", v_loss, global_steps)

    o, d, r, cum_ret= env.reset() / 255., False, 0, 0.
    mask = 0. if d else 1.
    h_t = torch.zeros(memory_size).cuda()
    o = o.reshape(*obs_dim)
    ep_ret = deque(maxlen=5)
    ep_ret.append(0.)

    explore_time=0
    train_time=0
    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):
        explore_start = time.time()
        ac_model.eval()

        for t in range(local_steps_per_epoch):
            with torch.no_grad():
                o_t = Tensor(o).cuda()
                mask_t = Tensor([mask]).cuda()
                a_t, logp_t, _, v_t, h_t_next = ac_model(o_t, mask_t, h_t)

            # save experience
            buf.store(o_t,a_t,r,v_t,logp_t,h_t,mask_t)

            h_t = h_t_next
            o, r, d, _ = env.step(a_t.data.item())
            o = (o/255.).reshape(*obs_dim)
            mask = 0. if d else 1.
            cum_ret += r

            if d or t == local_steps_per_epoch-1: # calculate the returns and GAE and reset environment
                if d:
                    buf.finish_path(0.)
                    ep_ret.append(cum_ret)
                    o, d, r, cum_ret= env.reset() / 255., False, 0, 0.
                    h_t = torch.zeros(memory_size).cuda()
                    o = o.reshape(*obs_dim)
                    mask = 0. if d else 1.
                else:
                    # environment does not end
                    with torch.no_grad():
                        o_t = Tensor(o).cuda()
                        mask_t = Tensor([mask]).cuda()
                        # x, _ = ac_model.process_feature(o_t, mask_t, h_t)
                        # last_val = ac_model.value_function(x)
                        _, _, _, last_val,_ = ac_model(ot,mask_t,h_t)
                    buf.finish_path(last_val)

        global_steps = (epoch + 1) * steps_per_epoch
        explore_stop = time.time()
        # Perform PPO update!
        ac_model.train()
        if batch_size != -1:
            update_with_batch()
        else:
            update()
        optim_stop = time.time()

        explore_time += explore_stop - explore_start
        train_time += optim_stop - explore_stop
        effective_fps = global_steps /(explore_time + train_time)
        env_fps = global_steps/explore_time
        
        return_sum = Tensor([sum(ep_ret)]).cuda()
        return_count = Tensor([len(ep_ret)]).cuda()
        return_sum = distributed_avg(return_sum)
        return_count = distributed_avg(return_count)

        avg_ret = return_sum.data.item()/return_count.data.item()	
        log_info(f"Episode({epoch}) Effective FPS: ({effective_fps:.2f}), Env FPS: ({env_fps:.2f}), avg return({avg_ret})")

        if rank == 0:
            writer.add_scalar("Return", avg_ret, global_steps)
            # Save model
            if epoch % save_freq == 0:
                torch.save(ac_model.state_dict(), "model_ppo.pt")



#add a global variable
world_size = 1
rank = 0

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--world-size',type=int, default=1)
    parser.add_argument('--rank',type=int,default=0)
    parser.add_argument('--local-rank',type=int,default=0)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=500)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--model-path', type=str, default=None)
    args = parser.parse_args()

    torch.multiprocessing.set_start_method('spawn')

    print("Collect Inputs...")
    # Distributed backend type
    dist_backend = 'nccl'
    # Url used to setup distributed training
    dist_url = "tcp://127.0.0.1:23456"

    print("Initialize Process Group...")
    # Initialize Process Group
    world_size = args.world_size
    rank = args.rank
    if world_size > 1:
        dist.init_process_group(backend=dist_backend, init_method=dist_url, rank=rank, world_size=world_size)
    # Establish Local Rank and set device on this node, i.e. the GPU index
    torch.cuda.set_device(args.local_rank)

    ppo(AI2ThorEnv,
        model=ActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid] * args.l),
        gamma=args.gamma,
        seed=args.seed,
        steps_per_epoch=args.steps,
        epochs=args.epochs,
        model_path=args.model_path)
