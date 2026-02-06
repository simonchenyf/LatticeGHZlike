"""
File: MaxQFI.py
Author: Yaofeng Chen, Xuanchen Zhang
Organization: Tsinghua University

Description:

    Used to obtain the maximal quantum Fisher information and the total
    time for reaching the maximum for different pulse seperation tau.

    This code produces Fig. 3b in the main text.

License:
MIT License

"""

import numpy as np
import torch
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
    
    device = 'cuda' # 'cpu'

    tau_range = 10 ** np.linspace(-2.5, -1.2, 50)
    Nbatch = 1000

    L = 20
    W = 20
    N = L * W
    alpha = 2.

    K = 2

    J0 = torch.ones((N, N), dtype=torch.float64, device=device) - torch.eye(N, dtype=torch.float64, device=device)
    i, j = np.meshgrid(np.arange(0,N,dtype=int), np.arange(0,N,dtype=int))
    dist = np.sqrt((i // W - j // W) ** 2 + (i % W - j % W) ** 2)
    dist = torch.eye(N, device=device) + torch.from_numpy(dist).to(device)
    J0 /= dist ** alpha

    chi = torch.matmul(J0, J0)
    chi -= torch.diag(torch.diag(chi))
    chi = torch.sum(chi) / (N*(N-1)*(N-2))

    opt_qfi_list = np.zeros_like(tau_range)
    time_opt_list = np.zeros_like(tau_range)
    qfi_x_list = np.zeros_like(tau_range)
    time_x_list = np.zeros_like(tau_range)

    for i, tau in enumerate(tau_range):

        Nperiod = int(1.5 * np.log(N) * 2 / tau ** 2 / chi / K ** 2 / N ** 2)

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


        for j in range(Nperiod):

            evo(state)
            
            all_pred[3*j+1] = get_pred(state)
            RotationX(state, -1)
            evo(state)
            RotationX(state, 1)
            all_pred[3*j+2] = get_pred(state)
            
            RotationY(state, -1)
            evo(state)
            RotationY(state, 1)
            all_pred[3*j+3] = get_pred(state)
        
        Sx = all_pred.cpu().numpy()[:,0]
        Sy = all_pred.cpu().numpy()[:,1]
        Sz = all_pred.cpu().numpy()[:,2]
        Sx2 = all_pred.cpu().numpy()[:,3]
        Sy2 = all_pred.cpu().numpy()[:,4]
        Sz2 = all_pred.cpu().numpy()[:,5]
        SxSy = all_pred.cpu().numpy()[:,6]
        SySz = all_pred.cpu().numpy()[:,7]
        SzSx = all_pred.cpu().numpy()[:,8]

        qfi_matrix = 4 * np.array([[Sx2 - Sx**2, SxSy - Sx*Sy, SzSx - Sx*Sz], 
                                [SxSy - Sx*Sy, Sy2 - Sy**2, SySz - Sy*Sz], 
                                [SzSx - Sx*Sz, SySz - Sy*Sz, Sz2 - Sz**2]])
    
        eigvals = np.linalg.eigvalsh(qfi_matrix.transpose(2,0,1))        # (3, 3*Nperiod+1)
        opt_qfi = eigvals[:,-1] / (N**2)
        max_opt_qfi = np.max(opt_qfi)
        time_opt = np.argmax(opt_qfi) * tau
            
        qfi_x = 4 * Sx2 / N ** 2
        max_qfi_x = np.max(qfi_x)
        time_x = np.argmax(qfi_x) * tau

        opt_qfi_list[i] = max_opt_qfi
        time_opt_list[i] = time_opt

        qfi_x_list[i] = max_qfi_x
        time_x_list[i] = time_x
    
    # fig 3b

    np.savetxt('./result/opt_qfi_tau.csv', np.stack((tau_range, opt_qfi_list), axis=1), delimiter=',')
    np.savetxt('./result/time_opt_tau.csv', np.stack((tau_range, time_opt_list), axis=1), delimiter=',')

    np.savetxt('./result/qfi_x_tau.csv', np.stack((tau_range, qfi_x_list), axis=1), delimiter=',')
    np.savetxt('./result/time_x_tau.csv', np.stack((tau_range, time_x_list), axis=1), delimiter=',')