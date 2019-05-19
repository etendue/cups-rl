import os, time
import torch
import numpy as np
from algorithms.ppo.ppo import PPO

import torch.distributed as dist
from tensorboardX import SummaryWriter


def dist_sum(x):
    dist.all_reduce(x, dist.ReduceOp.SUM)
    return x


def dist_mean(x):
    return dist_sum(x) / dist.get_world_size()


class TB_logger:
    def __init__(self, name, rank=0):
        self.rank = rank
        if self.rank == 0:
            self.writer = SummaryWriter(comment=name)

    def add_scalar(self, name, val, steps):
        if self.rank == 0:
            self.writer.add_scalar(name, val, steps)


def learner(model, rollout_storage, train_params, ppo_params, ready_to_works, queue, sync_flag, rank=0, distributed=False):
    '''
    learner use ppo algorithm to train model with experience from storage
    :param model:
    :param storage:
    :param params:
    :param ready_to_works:
    :param queue:
    :param sync_flag:
    :param rank:
    :return:
    '''
    print(f"learner with pid ({os.getpid()})  starts job")
    logger = TB_logger("ppo_ai2thor",rank)
    agent = PPO(actor_critic = model, **ppo_params)

    # start workers for next epoch
    _ = [e.set() for e in ready_to_works]
    # Training policy
    start_time = time.time()
    for epoch in range(train_params["epochs"]):
        rollout_ret = []
        rollout_steps = []
        # wait until all workers finish a epoch
        for _ in range(train_params["num_workers"]):
            rewards, steps = queue.get()
            rollout_ret.extend(rewards)
            rollout_steps.extend(steps)

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
        pi_loss, v_loss, kl, entropy = agent.update(rollout_storage)
        model.eval()

        # start workers for next epoch
        if epoch == train_params["epochs"] -1:
            # set exit flag to 1, and notify workers to exit
            sync_flag.value = 1

        _ = [e.set() for e in ready_to_works]

        # log statistics with TensorBoard
        ret_sum = np.sum(rollout_ret)
        steps_sum = np.sum(rollout_steps)
        episode_count = len(rollout_ret)

        if distributed:
            pi_loss = dist_mean(pi_loss)
            v_loss = dist_mean(v_loss)
            kl = dist_mean(kl)
            entropy = dist_mean(entropy)
            ret_sum = dist_sum(torch.tensor(ret_sum).cuda())
            steps_sum = dist_sum(torch.tensor(steps_sum).cuda())
            episode_count = dist_sum(torch.tensor(episode_count).cuda())

        # Log info about epoch
        global_steps = (epoch + 1) * train_params["steps"] * train_params["world_size"]
        fps = global_steps * train_params["world_size"] / (time.time() - start_time)
        print(f"Epoch [{epoch}] avg. FPS:[{fps:.2f}]")

        logger.add_scalar("KL", kl, global_steps)
        logger.add_scalar("Entropy", entropy, global_steps)
        logger.add_scalar("p_loss", pi_loss, global_steps)
        logger.add_scalar("v_loss", v_loss, global_steps)

        if episode_count > 0:
            ret_per_1000 = (ret_sum / steps_sum) * 1000
            logger.add_scalar("Return1000", ret_per_1000, global_steps)
            print(f"Epoch [{epoch}] Steps {global_steps}: "
                  f"return:({ret_per_1000:.1f})")
        else:
            print(f"Epoch [{epoch}] Steps {global_steps}: "
                  f"Goal is not reached in this epoch")

        if (epoch + 1) % 20 == 0 and rank == 0:
            if distributed:
                torch.save(model.module.state_dict(), f'model{epoch+1}.pt')
            else:
                torch.save(model.state_dict(), f'model{epoch+1}.pt')

    print(f"learner with pid ({os.getpid()})  finished job")