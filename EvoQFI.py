"""
File: EvoQFI.py
Author: Yaofeng Chen, Xuanchen Zhang
Organization: Tsinghua University

Description:

    Used to obtain the evolution of quantum Fisher information
    for different pulse seperation tau.

    This code produces Fig. 3a in the main text.

License:
MIT License

"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(1234)

def RotationX(state, direction):
    """
    direction = 1 or -1, indicating plus pi/2 flip and -pi/2 flip
    """
    state[[1,2]] = state[[2,1]]
    if direction == 1:
        state[1] *= -1
    if direction == -1:
        state[2] *= -1

def RotationY(state, direction):
    """
    direction = 1 or -1, indicating plus pi/2 flip and -pi/2 flip
    """
    state[[0,2]] = state[[2,0]]
    if direction == 1:
        state[2] *= -1
    if direction == -1:
        state[0] *= -1

def RotationZ(state, direction):
    """
    direction = 1 or -1, indicating plus pi/2 flip and -pi/2 flip
    """
    state[[0,1]] = state[[1,0]]
    if direction == 1:
        state[0] *= -1
    if direction == -1:
        state[1] *= -1

def get_pred(state):
    # [in_dim, site, batch]

    sx, sy, sz = torch.sum(state[0]/2, dim=0), torch.sum(state[1]/2, dim=0), torch.sum(state[2]/2, dim=0) # batch,
    avg_sx, avg_sy, avg_sz = torch.mean(sx, dim=-1), torch.mean(sy, dim=-1), torch.mean(sz, dim=-1)

    avg_sx_squared = torch.mean(sx ** 2, dim=-1)
    avg_sy_squared = torch.mean(sy ** 2, dim=-1)
    avg_sz_squared = torch.mean(sz ** 2, dim=-1)

    avg_sxsy = torch.mean(sx * sy, dim=-1)
    avg_sysz = torch.mean(sy * sz, dim=-1)
    avg_szsx = torch.mean(sz * sx, dim=-1)

    return torch.stack((avg_sx, avg_sy, avg_sz, avg_sx_squared, avg_sy_squared, avg_sz_squared, avg_sxsy, avg_sysz, avg_szsx), dim=-1)

if __name__ == "__main__":
    
    device = 'cuda'

    # We write code based on H = \frac{1}{2}\sum_{i\neq j}\sigma^z_i\sigma^z_j
    # So for eq(1) in main text we have K = 2

    tau = 0.025  # corresponding to K*tau = 0.05 since K = 2
    Nperiod = 48
    Nbatch = 1000

    L = 20
    W = 20
    N = L * W
    alpha = 2.

    J0 = torch.ones((N, N), dtype=torch.float64, device=device) - torch.eye(N, dtype=torch.float64, device=device)
    i, j = np.meshgrid(np.arange(0,N,dtype=int), np.arange(0,N,dtype=int))
    dist = np.sqrt((i // W - j // W) ** 2 + (i % W - j % W) ** 2)
    dist = torch.eye(N, device=device) + torch.from_numpy(dist).to(device)
    J0 /= dist ** alpha

    sz = torch.ones((N, Nbatch), dtype=torch.float64, device=device)
    crit = 1
    while crit > 1e-5:
        sx = 2.0 * torch.randint(0,2, size=(N, Nbatch), dtype=torch.float64, device=device) - 1.0
        crit = torch.abs(torch.mean(sx))

    crit = 1
    while crit >1e-5:
        sy = 2.0 * torch.randint(0,2, size=(N, Nbatch), dtype=torch.float64, device=device) - 1.0
        crit = torch.abs(torch.mean(sy))

    y0 = torch.stack((sx, sy, sz), dim=0) # 3, N, Nbatch
    del sx, sy, sz

    def evo(state):
        theta = 2 * tau * J0.matmul(state[2])
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)

        sx = state[0] * cos_theta - state[1] * sin_theta
        sy = state[0] * sin_theta + state[1] * cos_theta

        state[0] = sx
        state[1] = sy

    # to keep y0 in case of future use
    state = torch.zeros_like(y0)
    state[:] = y0[:]

    all_pred = torch.zeros((3*Nperiod+1, 9), dtype=state.dtype, device=state.device) # 3*Nperiod+1, 9
    all_pred[0] = get_pred(state)


    for i in range(Nperiod):

        evo(state)
        
        all_pred[3*i+1] = get_pred(state)
        RotationX(state, -1)
        evo(state)
        RotationX(state, 1)
        all_pred[3*i+2] = get_pred(state)
        
        RotationY(state, -1)
        evo(state)
        RotationY(state, 1)
        all_pred[3*i+3] = get_pred(state)
    
    all_pred = all_pred.cpu().numpy()

    qfi = 4 * (all_pred[:,3] - all_pred[:,0]**2) / N ** 2

    t = np.linspace(0, 3*Nperiod*tau, 3*Nperiod+1)

    # fig 3a with K\tau = 0.05
    qfi_data = np.stack((t, qfi), axis=1)
    np.savetxt('./result/QFI_5.csv', qfi_data, delimiter=',')
