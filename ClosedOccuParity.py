"""
File: ClosedOccuParity.py
Author: Yaofeng Chen, Xuanchen Zhang
Organization: Tsinghua University

Description:

    Lanczos-based implementation for obtaining the end state population
    and parity oscillation in Figs. 1 (a1), (a2), (b1) and (b2).

Make sure you run the following codes before running this code:
    complieC.py
    ClosedEvo.py
        
License:
MIT License

"""

import numpy as np
import getSz, zeroize, subLanczos
from scipy.linalg import eigh_tridiagonal

class LanczosFlip:
    
    def __init__(self, J0, m, bound):
        self.m = m
        self.bound = bound
        self.x = np.zeros((m,), dtype=np.float64)
        self.y = np.zeros((m-1,), dtype=np.float64)
        self.J0 = J0
        self.N = J0.shape[0]
        self.dim = 2**self.N
        self.V = np.zeros((m, self.dim),dtype=np.complex128)
        self.v1 = np.zeros((self.dim,),dtype=np.complex128)
        self.w = np.zeros((self.dim,),dtype=np.complex128)
        self.V[-1,-1]+=1
        self.V[-1,-1]-=1

    def flip(self, v0, alongx, theta):

        N = self.N
        m = self.m
        bound = self.bound
        x = self.x
        y = self.y
        V = self.V
        v1 = self.v1
        w = self.w

        x[:] = 0
        y[:] = 0
        zeroize.apply(N, m, V, v1, w)

        beta = np.linalg.norm(v0)

        for i in range(m):

            x[i] =  subLanczos.apply(N, beta, v0, v1, w, V[i], alongx)

            beta = np.linalg.norm(w)

            if i < (m-1):
                y[i] = beta
            if beta < bound:
                break

            v0, v1, w = w, v0, v1

        val, vec = eigh_tridiagonal(x[:i+1],y[:i])

        U = (vec.T).dot(V[:i+1,:])

        psi_dt = (np.exp(-1.j*val*theta) * np.conj(vec[0, :].T)).dot(U)

        return psi_dt

# ------------------------  switch for 1d and 2d below ------------------------

data = np.load("./data/GHZlikePsi_1d_L20_a1_tau005.npz") # for 1d
# data = np.load("./data/GHZlikePsi_2d_L4_a3_tau003.npz") # for 2d

# ------------------------  switch for 1d and 2d above ------------------------

N = data['N']
psi = data['psi']

spin_up_count = np.zeros(len(psi))
ref = np.arange(0,1<<N,dtype=int)
for i in range(N):
    spin_up_count += (np.bitwise_and(ref, 1<<i)>>i)

lanczos = LanczosFlip(np.zeros((N,N), dtype=np.float64), m=20, bound=1e-12)
Sz = np.zeros((len(psi),), dtype=np.float64)
getSz.apply(N, Sz)

psi = lanczos.flip(psi, False, np.pi/4) # rotate x -> z

# get parity
def get_parity(psi, theta, spin_up_count):
    psi = psi * np.exp(1.j * theta / 2 * Sz) # Sz is sum of sigma z
    psi = lanczos.flip(psi, False, np.pi/4) # rotate about y
    prob = np.abs(psi) ** 2
    parity = (-1) ** spin_up_count.astype(int)
    print(np.sum(prob * parity))
    return np.sum(prob * parity)

theta = np.linspace(np.pi*(0.-0.5), np.pi*(0.+0.5),151)
parity = np.zeros_like(theta)
for i, a in enumerate(theta):
    parity[i] = get_parity(psi, a, spin_up_count)

# fig 1 a2 b2
np.save('./result/parity_1d_L20_a1_tau005.npy', parity)
# np.save('./result/parity_2d_L4_a3_tau003.npy', parity)

# get occupation
prob = np.abs(psi)**2
ind = np.argsort(spin_up_count)
count_prob = prob[ind]
_, unique_ind = np.unique(spin_up_count[ind], return_index=True)
occupation = np.zeros((len(unique_ind),), dtype=np.float64)
for i in range(len(occupation)-1):
    occupation[i] = np.sum(count_prob[unique_ind[i]:unique_ind[i+1]])
occupation[-1] = np.sum(count_prob[unique_ind[-1]:])
print(occupation)

# fig 1 a1 b1
np.save('./result/occupation_1d_L20_a1_tau005.npy', occupation)
# np.save('./result/occupation_2d_L4_a3_tau003.npy', occupation)
