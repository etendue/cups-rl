import torch
def reset(env, state_size):
    o= env.reset()
    mask_t = torch.tensor(0.,dtype=torch.float32).cuda()
    prev_a = torch.tensor(0, dtype=torch.long).cuda()
    obs_t = torch.Tensor(o/255.).cuda().unsqueeze(dim=0)  # 128x128 -> 1x128x128
    state_t = torch.zeros(state_size, dtype=torch.float32).cuda()
    x = {"observation":obs_t,
        "memory":{
            "state":state_t,
            "mask":mask_t,
            "action": prev_a
        }}
    return x