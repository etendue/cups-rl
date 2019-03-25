import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
import time
import scipy.signal
from algorithms.ppo.core import ActorCritic, count_vars,average_gradients, \
    sync_all_params,mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs
from  gym_ai2thor.envs.ai2thor_env import AI2ThorEnv
from tensorboardX import SummaryWriter


def log_info(msg):
    if proc_id() == 0:
        print(msg)


class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, memory_size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(
            self._combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(
            self._combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.h_buf =np.zeros((size, memory_size), dtype=np.float32)
        self.mask_buf = np.zeros(size, dtype=np.float32)
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
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = self._discount_cumsum(
            deltas, self.gamma * self.lam)

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
        adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        return [
            self.obs_buf, self.act_buf, self.adv_buf, self.ret_buf,
            self.logp_buf, self.h_buf, self.mask_buf
        ]

    def _combined_shape(self, length, shape=None):
        if shape is None:
            return (length, )
        return (length, shape) if np.isscalar(shape) else (length, *shape)

    def _discount_cumsum(self, x, discount):
        """
        magic from rllab for computing discounted cumulative sums of vectors.

        input:
            vector x,
            [x0,
            x1,
            x2]

        output:
            [x0 + discount * x1 + discount^2 * x2,
            x1 + discount * x2,
            x2]
        """
        return scipy.signal.lfilter([1], [1, float(-discount)],
                                    x[::-1],
                                    axis=0)[::-1]


"""

Proximal Policy Optimization (by clipping),

with early stopping based on approximate KL

"""


def ppo(env_fn,
        actor_critic=ActorCritic,
        ac_kwargs=dict(),
        seed=0,
        steps_per_epoch=4000,
        epochs=50,
        gamma=0.99,
        clip_ratio=0.2,
        lr=3e-4,
        train_iters=1,
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

        actor_critic: The agent's main model which is composed of
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
    horizon_t = 32

    seed += 10000 * proc_id()
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = env_fn()
    obs_dim = env.observation_space.shape
    act_dim = 1

    # Share information about action space with policy architecture
    ac_kwargs['action_space'] = env.action_space

    # Main model
    actor_critic = actor_critic(in_features=obs_dim, **ac_kwargs)

    if model_path:
        actor_critic.load_state_dict(torch.load(model_path))

    if torch.cuda.is_available():
        device = torch.device('cuda')
        actor_critic.to(device)

    # Experience buffer
    local_steps_per_epoch = int(steps_per_epoch / num_procs())
    buf = PPOBuffer(obs_dim, act_dim, local_steps_per_epoch, memory_size, gamma, lam)

    # Count variables
    var_counts = tuple(
        count_vars(module)
        for module in [actor_critic.policy_, actor_critic.value_function_, actor_critic.feature_base_])
    log_info('\nNumber of parameters: \t pi: %d, \t v: %d \tbase: %d\n' % var_counts)

    # Optimizers
    optimizer = torch.optim.Adam(actor_critic.parameters(), lr=lr)

    # Sync params across processes
    sync_all_params(actor_critic.parameters())

    # add TensorBoard
    if proc_id() == 0:
        writer = SummaryWriter(comment="ai2thor_pp")

    def update():
        obs, act, adv, ret, logp_old, h, mask = [torch.Tensor(x) for x in buf.get()]

        # Training policy
        for i in range(train_iters):
            # Output from policy function graph
            x = actor_critic.process_feature(obs,mask,h,horizon_t)
            _, logp_a, ent = actor_critic.policy(x)
            v = actor_critic.value_function(x)
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
            kl = mpi_avg(kl.item())
            if kl > 1.5 * target_kl:
                log_info('Early stopping at step %d due to reaching max kl.' % i)
                break

            # Policy gradient step
            optimizer.zero_grad()
            (pi_loss + v_loss * alpha - entropy*beta).backward()
            average_gradients(optimizer.param_groups)
            optimizer.step()

        # Log info about epoch TODO
        if proc_id() == 0:
            global_steps = epoch * steps_per_epoch
            writer.add_scalar("Entropy", entropy, global_steps)
            writer.add_scalar("p_loss", pi_loss, global_steps)
            writer.add_scalar("v_loss", v_loss, global_steps)

    start_time = time.time()

    o, d, cum_ret, h = env.reset() / 255., False, 0, np.zeros(memory_size, dtype=float)
    mask = 0. if d else 1.
    o = o.reshape(*obs_dim)
    ep_ret = 0
    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):
        epoch_start = time.time()
        actor_critic.eval()

        for t in range(local_steps_per_epoch):
            with torch.no_grad():
                a_t, logp_t, _, v_t, h = actor_critic(Tensor(o), Tensor([mask]), Tensor(h))

            # save experience
            buf.store(o,
                      a_t.cpu().data.numpy(),
                      r,
                      v_t.cpu().data.numpy(),
                      logp_t.cpu().data.numpy(),
                      h.cpu().data.numpy(),
                      mask)

            o, r, d, _ = env.step(a_t.cpu().data.numpy().item())
            o = (o/255.).reshape(*obs_dim)
            mask = 0. if d else 1.
            cum_ret += r

            if d: # calculate the returns and GAE and reset environment
                buf.finish_path(0.)
                ep_ret = cum_ret
                o, d, cum_ret, h = env.reset() / 255., False, 0, np.zeros(memory_size, dtype=float)
                o = o.reshape(*obs_dim)
                mask = 0. if d else 1.

        # environment does not end
        if not d:
            with torch.no_grad():
                x, _ = actor_critic(Tensor(0),Tensor(mask), Tensor(h))
                last_val = actor_critic.value_function(x).cpu().data.numpy()[0]
            buf.finish_path(last_val)

        eval_stop = time.time()
        # Perform PPO update!
        actor_critic.train()
        update()
        opti_stop = time.time()

        train_time = (opti_stop - eval_stop)/(opti_stop-epoch_start)
        fps = (epoch+1)*steps_per_epoch//(opti_stop - start_time)
        log_info(f"FPS: ({fps}), training time ({int(train_time*100)})%")

        epoch_ret = mpi_avg(ep_ret)
        if proc_id() == 0:
            global_steps = epoch * steps_per_epoch
            writer.add_scalar("Return", epoch_ret, global_steps)
            # Save model
            if epoch % save_freq == 0:
                torch.save(actor_critic.cpu().state_dict(),"model/ppo.pt")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=1)
    parser.add_argument('--steps', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='ppo')
    parser.add_argument('--model-path', type=str, default=None)
    args = parser.parse_args()

    mpi_fork(args.cpu)  # run parallel code with mpi


    ppo(AI2ThorEnv,
        actor_critic=ActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid] * args.l),
        gamma=args.gamma,
        seed=args.seed,
        steps_per_epoch=args.steps,
        epochs=args.epochs,
        model_path=args.model_path)
