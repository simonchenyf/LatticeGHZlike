"""
File: util.py
Author: Yaofeng Chen, Xuanchen Zhang
Organization: Tsinghua University

Description:

    Backend code. Define the GHZlike class used for sweeping parameters.

License:
MIT License

"""

import torch
import numpy as np
import matplotlib.pyplot as plt

# Fig. 3(a1, b1)

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


class GHZlike:
    def __init__(self, Nbatch, ratio, test_name='default', device='cuda'):

        self.device = device
        self.dtype = torch.float64
        self.Nbatch = Nbatch
        self.ratio = ratio

        self.test_name = test_name
        self.dir_path =  './data' + test_name

        self.pred_label = [r'$<s_x>$', r'$<s_y>$', r'$<s_z>$', r'$<s_x^2>$', r'$<s_y^2>$', r'$<s_z^2>$', r'$<s_xs_y>_\text{sym}$', r'$<s_ys_z>_\text{sym}$', r'$<s_zs_x>_\text{sym}$']

    def set_alpha(self, alpha):
        self.alpha = alpha
        self.K = 2

    def set_config(self, L, W):

        self.L = L
        self.W = W
        self.N = L * W

        sz = torch.ones((self.N, self.Nbatch), dtype=self.dtype, device=self.device)

        crit = 1
        while crit > 1e-5:
            sx = 2.0 * torch.randint(0,2, size=(self.N, self.Nbatch), dtype=self.dtype, device=self.device) - 1.0
            crit = torch.abs(torch.mean(sx))

        crit = 1
        while crit >1e-5:
            sy = 2.0 * torch.randint(0,2, size=(self.N, self.Nbatch), dtype=self.dtype, device=self.device) - 1.0
            crit = torch.abs(torch.mean(sy))

        self.state0 = torch.stack((sx, sy, sz), dim=0) # 3, N, Nbatch

        J0 = torch.ones((self.N, self.N), dtype=self.dtype, device=self.device) - torch.eye(self.N, dtype=self.dtype, device=self.device)
        i, j = np.meshgrid(np.arange(0,self.N,dtype=int), np.arange(0,self.N,dtype=int))
        dist = np.sqrt((i // self.W - j // self.W) ** 2 + (i % self.W - j % self.W) ** 2)
        dist = torch.eye(self.N, device=self.device) + torch.from_numpy(dist).to(self.device)
        J0 /= dist ** self.alpha

        self.J0 = J0

        chi = torch.matmul(self.J0, self.J0)
        chi -= torch.diag(torch.diag(chi))
        chi = torch.sum(chi) / (self.N*(self.N-1)*(self.N-2))
        self.chi = chi.item()

    def set_tau(self, tau):
        self.tau = tau
        self.Nperiod = int(3 * np.log(self.N) / (self.chi * (self.tau * self.N * self.K) ** 2)) + 1
        self.all_pred = torch.zeros((3*self.Nperiod+1, 9), dtype=self.dtype, device=self.device)
        self.epsilon = self.N * self.chi * self.K * self.tau / 2 # alpha

    def set_all(self, L, W, Nperiod, tau, alpha):
        self.set_config(L, W)
        self.set_alpha(alpha)
        self.set_Nperiod(Nperiod)
        self.set_tau(tau)

    def step(self, state):
        theta = 2 * self.tau * self.J0.matmul(state[2])
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)

        sx = state[0] * cos_theta - state[1] * sin_theta
        sy = state[0] * sin_theta + state[1] * cos_theta

        state[0] = sx
        state[1] = sy


    def evo(self):
        state = torch.zeros_like(self.state0)
        state[:] = self.state0[:]
        self.all_pred[0] = get_pred(state)

        for i in range(self.Nperiod):
            RotationX(state, -1)
            self.step(state)
            RotationX(state, 1)
            self.all_pred[3*i+1] = get_pred(state)
            RotationY(state, -1)
            self.step(state)
            RotationY(state, 1)
            self.all_pred[3*i+2] = get_pred(state)
            self.step(state)
            self.all_pred[3*i+3] = get_pred(state)

        self.qfi = (self.all_pred[:,3] - self.all_pred[:,0]**2) * 4 / self.N**2

    def save(self):

        path = self.dir_path + f'/alpha={np.round(self.alpha,2)}/L={int(self.L)}/tau={np.round(self.tau,6)}/dtwa.pt'
        data = {
            "L": self.L,
            "W": self.W,
            "alpha": self.alpha,
            "tau": self.tau,
            "Nperiod": self.Nperiod,
            "pred": self.all_pred,
            "qfi": self.qfi
        }

        torch.save(data, path)

    def plot(self):
        path = self.dir_path + f'/alpha={np.round(self.alpha,2)}/L={int(self.L)}/tau={np.round(self.tau,6)}/'

        t = np.arange(0, len(self.all_pred), 1) * self.tau
        pred = self.all_pred.cpu().numpy()
        qfi = self.qfi.cpu().numpy()

        plt.plot(self.effective_t * (6 / self.N / self.chi / self.K**2 / self.tau), self.FQ_eff, label='effective')
        plt.plot(t, qfi, label='DTWA')
        plt.title(f'L={int(self.L)}, W={int(self.W)}, alpha={np.round(self.alpha,2)}, Nperiod={int(self.Nperiod)}, Nbatch={int(self.Nbatch)},\n chi={np.round(self.chi, 6)}, tau={np.round(self.tau,6)},epsilon={np.round(self.epsilon,6)}')
        plt.ylabel(r'$F_Q/N^2$')
        plt.savefig(path+'qfi.png')
        plt.close()

        fig = plt.figure(figsize=(12,12))
        fig.suptitle(f'L={int(self.L)}, W={int(self.W)}, alpha={np.round(self.alpha,2)}, Nperiod={int(self.Nperiod)}, Nbatch={int(self.Nbatch)},\n chi={np.round(self.chi, 6)}, tau={np.round(self.tau,6)},epsilon={np.round(self.epsilon,6)}')
        for i in range(9):
            ax = fig.add_subplot(3,3,i+1)
            ax.plot(t, pred[:,i])
            ax.set_title(self.pred_label[i])
        plt.savefig(path+'pred.png')
        plt.close()

    def run(self):
        self.evo()
        self.save()
        self.plot()

    def effective(self, theta=np.pi/2, phi=0):

        J = self.N / 2
        Tc = 1.5 * np.log(self.N) / self.N

        t = np.linspace(0, Tc, 101)

        Sz_diag = np.arange(J, -J-1, -1.).astype(np.complex128)
        Sx_offdiag = np.sqrt(J*(J+1)-Sz_diag[1:]*(Sz_diag[1:]+1)).astype(np.complex128)
        Sx = (np.diag(Sx_offdiag, 1) + np.diag(Sx_offdiag, -1))/2
        Sy = (np.diag(Sx_offdiag, 1) - np.diag(Sx_offdiag, -1))/2j
        Sz = np.diag(Sz_diag)

        psi0 = np.zeros((self.N+1,), dtype=np.complex128)
        psi0[0] = 1

        Hxyz = (2/self.N) * (np.dot(Sx, np.dot(Sy, Sz)) + np.dot(Sz, np.dot(Sy, Sx)))
        
        eigval, eigvec = np.linalg.eigh(Hxyz)

        # Nt, dim, 1
        psi_t = np.matmul(eigvec, np.expand_dims(np.exp(-1j*eigval*t.reshape(-1,1)) * np.dot(np.conj(eigvec.T), psi0), axis=2))

        self.effective_t = t
        Sx_psi = np.matmul(Sx,psi_t) # Nt, dim, 1
        self.FQ_eff = 4 * ( np.linalg.norm(Sx_psi.squeeze(), axis=1)**2 - np.matmul(np.conj(psi_t.transpose(0,2,1)), Sx_psi).squeeze() ) / self.N**2

        np.savez(self.dir_path + f'/alpha={np.round(self.alpha,2)}/L={int(self.L)}/effective.npz', t=t, FQ_eff=self.FQ_eff)

if __name__ == "__main__":

    test = GHZlike(Nbatch=1024, ratio=0.8, device='cuda')

    test.set_alpha(6.0)

    test.set_config(32,1)
    test.effective()

    test.set_tau(0.00631)

    test.run()