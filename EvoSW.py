"""
File: EvoSW.py
Author: Xuanchen Zhang, Yaofeng Chen
Organization: Tsinghua University

Description:

    Used to obtain the evolution of spin wave excitation
    for different pulse seperation tau.

    This code produces the inset of Fig. 2a in the main text.

License:
MIT License

"""

import numpy as np
import matplotlib.pyplot as plt

L = 20
alpha = 2.0

tau = 0.04 # corresponding to K*tau = 0.08 since K = 2
Nperiod = 19
t = np.linspace(0, 3*Nperiod*tau, 3*Nperiod+1) # (3*Nperiod+1,)

N = L ** 2

q1, q2 = np.meshgrid(np.arange(0, L, dtype=int), np.arange(0, L, dtype=int))
q = 2 * np.pi * np.stack((q1.reshape(-1), q2.reshape(-1)), axis=-1) / L   # (N, 2)

r1, r2 = np.meshgrid(np.arange(-L/2, L/2, dtype=int), np.arange(-L/2, L/2, dtype=int))
r = np.stack((r1.reshape(-1), r2.reshape(-1)), axis=-1)   # (N, 2)
r_length = np.linalg.norm(r, axis=1)    # (N,)
r_length[(L+1)*int(L/2)] = 1

phase = np.exp(-1j * np.dot(q, r.T))    # (N, N)
phase[:,(L+1)*int(L/2)] *= 0

Kq = np.real(np.sum(phase / (r_length ** alpha), axis=-1))    # (N,)
Aq = Kq[:1] - Kq[1:]  # (N-1,)

r_zero = np.ones_like(r_length)
r_zero[(L+1)*int(L/2)] = 0
T02 = np.real(np.sum(r_zero / (r_length ** (2*alpha)), axis=-1))
Bq = (Kq[1:] ** 2 - T02) / 2 # (N-1,)

epsilon_q = np.sqrt(Aq ** 2 - (tau ** 2) * (Bq ** 2))  # (N-1,)
UVq = (tau ** 2) * (Bq ** 2) / (2 * (Aq ** 2 - (tau ** 2) * (Bq ** 2)))  # (N-1,)

Nfm = np.sum(UVq * (1 - np.cos(epsilon_q * t.reshape(-1,1))), axis=1) # (3*Nperiod+1,)

plt.plot(t, Nfm)
plt.savefig('SpinWave_Excitation_8.png')

Nfm_data = np.stack((t, Nfm), axis=-1)
np.savetxt('SpinWave_Excitation_8.csv', Nfm_data, delimiter=',')