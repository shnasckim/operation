import torch
import torch.nn as nn


def torch_pool(inputs, target_size):
    print(inputs.size())
    print("a", torch.arange(target_size, dtype=torch.float32))
    print("b", (inputs.size(-1) / target_size), 10/7)
    print("c",  (torch.arange(target_size, dtype=torch.float32) * (inputs.size(-1) / target_size)))
    start_points = (torch.arange(target_size, dtype=torch.float32) * (inputs.size(-1) / target_size)).long()
    print('start_points :', start_points)
    end_points = ((torch.arange(target_size, dtype=torch.float32)+1) * (inputs.size(-1) / target_size)).ceil().long() 
    print("d", ((torch.arange(target_size, dtype=torch.float32)+1) * (inputs.size(-1) / target_size)))
    print('end_points :', end_points)
    pooled = []
    for idx in range(target_size):
        pooled.append(torch.mean(inputs[:, :, start_points[idx]:end_points[idx]], dim=-1, keepdim=False))
    pooled = torch.cat(pooled, -1)
    return pooled


import numpy as np
# inps = np.array([0, 1, 2, 3, 4, 5, 6], dtype=np.float32)[None, :, None]
inps = np.array([0, 1, 2, 3, 4, 5, 6,7,8,9], dtype=np.float32)[None, :, None]
inps_torch = np.transpose(inps, (0, 2, 1))
print(inps_torch, inps_torch.shape)
x = torch_pool(torch.tensor(inps_torch), 7)
print(x)
