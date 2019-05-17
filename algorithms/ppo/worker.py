import os
import torch
from  gym_ai2thor.envs.ai2thor_env import AI2ThorEnv
from algorithms.ppo.utils import reset

def worker(policy,
           storage,
           start_event,
           queue,
           exit_flag,
           gpu_id=0,
           display=None,
           task_config_file="config_files/default.json"):
    '''
    Worker function to collect experience based on policy and store the experience in storage
    param policy: function/actor-critic
    storage: 
    start_event: event to synchronize work and training
    queue: message queue to send episode reward to learner
    exit_flag: flag set by leaner to exit the job
    gpu_id: gpud id where cuda is running on
    display: where environment is running one, if None, it is set as gpu_id
    task_config_file: the task configuration file
    '''
    print(f"Worker with pid ({os.getpid()}) starts ...")
    
    torch.cuda.set_device(gpu_id)
    if display is None:
        os.environ['DISPLAY'] = f':{gpu_id}'
    else:
        os.environ['DISPLAY'] = display

    steps_per_epoch = storage.shape[0]
    state_size = storage.h_buf.shape[1]
    
    env = AI2ThorEnv(config_file=task_config_file)
    x = reset(env, state_size)
    episode_reward, episode_steps = 0., 0

    # Wait for start job
    start_event.wait()
    while exit_flag.value != 1:
        for i in range(steps_per_epoch):
            with torch.no_grad():
                a_t, logp_t, _, v_t, state_t = policy(x)
                # interact with environment
                o, r, d, _ = env.step(a_t.item())
                episode_reward += r  # accumulate reward within one rollout.
                episode_steps += 1
                r_t = torch.tensor(r, dtype=torch.float32).cuda()
                # save experience
                storage.store(action=a_t, 
                              reward=r_t, 
                              value=v_t, 
                              logp=logp_t,
                              state=x["memory"]["state"],
                              mask=x["memory"]["mask"])
                # prepare inputs for next step
                x["observation"] = torch.Tensor(o/255.).cuda().unsqueeze(dim=0)  # 128x128 -> 1x128x128
                x["action"] = a_t
                x["memory"]["state"] = state_t
                x["memory"]["mask"] = torch.tensor((d+1)%2, dtype=torch.float32).cuda()
                x["memory"]["action"] = a_t
                # check terminal state
                epoch_end = (i == (steps_per_epoch - 1))
                if d: # calculate the returns and GAE and reset environment
                    storage.finish_path(0)
                    msg = {"msg_type":"train", 
                            "reward":episode_reward, 
                            "steps":episode_steps, 
                            "epoch_end":epoch_end}
                    queue.put(msg)
                    x = reset(env, state_size)
                    episode_reward, episode_steps = 0., 0
                elif epoch_end: # early cut due to reach maximum steps in on epoch
                    _, _, _, last_val, _ = policy(x)
                    storage.finish_path(last_val)
                    episode_reward += last_val.item()
                    msg = {"msg_type":"train",
                           "reward":None,
                           "steps":episode_steps,
                           "epoch_end":epoch_end}
                    queue.put(msg)
                    # x = reset(env, state_size)
                    # episode_reward, episode_steps = 0., 0
        # Wait for next job
        start_event.clear()
        start_event.wait()

    print(f"Worker with pid ({os.getpid()})  finished job")