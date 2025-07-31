"""
File: ClosedEvo.py
Author: Yaofeng Chen, Xuanchen Zhang
Organization: Tsinghua University

Description:

    Lanczos-based evolution for obtaining the end state in a closed system,
    used to calculate end state population and parity oscillation in Fig. S1
    in the Supplementary Information.

Make sure you run the following codes before running this code:
    complieC.py

Next file to use: ClosedOccuParity.py

License:
MIT License

"""

import numpy as np
import matplotlib.pyplot as plt
import getJzz
from util import Lanczos_unstable, Prediction
import os
os.environ["MKL_NUM_THREADS"] = "152"

# ------------------------  switch for 1d and 2d below ------------------------

# for 1d
tau = 0.05
alpha = 1.
L = 20
W = 1

# # for 2d
# tau = 0.03
# alpha = 3.
# L = 4
# W = 4

# ------------------------  switch for 1d and 2d above ------------------------

N = L * W
Nperiod = 22

m = 20
bound = 1e-12

psi0 = np.zeros((2**N,), dtype=np.complex128)
psi0[0] = 1.

sz = np.array([1.,-1.], dtype=np.complex128)
iden = np.array([1.,1.], dtype=np.complex128)

J0 = np.ones((N, N), dtype=np.float64) - np.eye(N, dtype=np.float64)    
for i in range(N):
    for j in range(N):
        if i != j:
            dist = np.sqrt((i // W - j // W) ** 2 + (i % W - j % W) ** 2)
            J0[i,j] /= dist ** alpha

lanczos = Lanczos_unstable(J0, m=m, bound=bound)
prediction = Prediction(N, np.complex128)

Jzz = np.zeros((2**N,), dtype=psi0.dtype)
getJzz.apply(N, tau, J0, Jzz)
Rzz = np.exp(Jzz)

psi = psi0

all_pred = np.zeros((3*Nperiod+1, 9), dtype=np.float64)
all_pred[0] = prediction.get_pred(psi)

for i in range(Nperiod):
    # True for Sx, False for Sy
    psi = lanczos.flip(psi, True, -np.pi/4)
    psi *= Rzz
    psi = lanczos.flip(psi, True, np.pi/4)
    all_pred[3*i+1,:] = prediction.get_pred(psi)
    psi = lanczos.flip(psi, False, -np.pi/4)
    psi *= Rzz
    psi = lanczos.flip(psi, False, np.pi/4)
    all_pred[3*i+2,:] = prediction.get_pred(psi)
    psi *= Rzz
    all_pred[3*i+3,:] = prediction.get_pred(psi)

np.savez("./data/GHZlikePsi_1d_L20_a1_tau005.npz", N=N, psi=psi)
# np.savez("./data/GHZlikePsi_2d_L4_a3_tau003.npz", N=N, psi=psi)