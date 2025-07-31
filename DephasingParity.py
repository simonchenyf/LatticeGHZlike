"""
File: DephasingParity.py
Author: Yaofeng Chen, Xuanchen Zhang
Organization: Tsinghua University

Description:

    This code obtains the parity oscillation of the obtained end states
    for different dephasing strengh Gamma.
    
    This code produces Fig. 4b in the main text.

Make sure you run the following codes before running this code:
    Dephasing.py

License:
MIT License

"""

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np

N = 12

sigma = cp.array([[0, -1j], [1j, 0]], dtype=cp.complex128)
val, vec = cp.linalg.eigh(sigma)
ry = vec.dot(cp.diag(cp.exp(-1j*cp.pi/4*val)).dot(cp.conj(vec.T)))

Ry = cp.array(1., dtype=cp.complex128)
for _ in range(N):
    Ry = cp.kron(Ry, ry)
RyT = cp.conj(Ry.T)

iden = cp.ones(2, dtype=cp.complex128)
sz = cp.array([1.,-1.], dtype=cp.complex128)

Sz = cp.zeros((2**N,), dtype=cp.complex128)
for i in range(N):
    new = cp.array(1., dtype=cp.complex128)
    for j in range(i):
        new = cp.kron(new, iden)
    new = cp.kron(new, sz)
    for j in range(i+1, N):
        new = cp.kron(new, iden)
    Sz += new

rho = cp.load('./data/opensystem/endstate_dephasing.npy')

rho = Ry.dot(rho.dot(RyT))  # rotate x-> z

spin_up_count = cp.zeros(1<<N)
ref = cp.arange(0,1<<N,dtype=int)
for i in range(N):
    spin_up_count += (cp.bitwise_and(ref, 1<<i)>>i)

# get parity
def get_parity(rho, theta, spin_up_count):
    rho = cp.exp(1.j * theta / 2 * Sz).reshape(-1,1) * rho * cp.exp(-1.j * theta / 2 * Sz).reshape(1,-1) # Sz is sum of sigma z
    rho = Ry.dot(rho.dot(RyT))
    prob = cp.diag(rho).get().real
    parity = (-1) ** spin_up_count.astype(int)
    return np.sum(prob * parity)

theta = np.linspace(cp.pi*(-0.5), cp.pi*0.5, 151)
parity = np.zeros_like(theta)
for i, a in enumerate(theta):
    parity[i] = get_parity(rho, a, spin_up_count)

np.save('./result/parity_dephasing.npy', parity)

np.savetxt('./result/parity_dephasing.csv', np.stack((theta, parity), axis=1))

plt.plot(theta, parity)
plt.savefig('./result/parity_dephasing.png')