"""
File: Dephasing.py
Author: Yaofeng Chen, Xuanchen Zhang
Organization: Tsinghua University

Description:

    This code obtains the parity of endstate
    for local and global dephasing.
    
    This code produces the inset of Fig. 5.

License:
MIT License

"""

import dynamiqs as dq
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import functools
import cupy as cp

dq.set_device('gpu', index=0)
dq.set_precision('double')

N = 12

state_list = [dq.fock_dm(2, 0)] * N
rho = dq.tensor(*state_list).asdense()

tau = 0.18/560

Gamma = 10

# ------------------------  switch for local and global dephasing below ------------------------

# local dephasing
Nperiod = 7
c_ops = [sz_list[i].asdense() * np.sqrt(Gamma)/2 for i in range(N)]

# # global dephasing
# Nperiod = 5
# Sz_total = sz_list[0]
# for i in range(N-1):
#     Sz_total += sz_list[i+1]
# c_ops = [Sz_total.asdense() * sqrt_Gamma_z/2]

# ------------------------  switch for local and global dephasing above ------------------------

alpha = 1.

J0 = 560 * (np.ones((N, N), dtype=np.float64) - np.eye(N, dtype=np.float64))   
for i in range(N):
    for j in range(N):
        if i != j:
            dist = np.abs(i - j)
            J0[i,j] /= dist ** alpha


sx_list, sy_list, sz_list = [], [], []
for i in range(N):
    op_list = [dq.eye(2)] * N
    op_list[i] = dq.sigmax()
    sx_list.append(dq.tensor(*op_list).asdense())
    op_list[i] = dq.sigmay()
    sy_list.append(dq.tensor(*op_list).asdense())
    op_list[i] = dq.sigmaz()
    sz_list.append(dq.tensor(*op_list).asdense())


Rx = dq.tensor(*([dq.rx(-np.pi/2)] * N)).asdense()
Ry = dq.tensor(*([dq.ry(-np.pi/2)] * N)).asdense()
RxT = dq.tensor(*([dq.rx(np.pi/2)] * N)).asdense()
RyT = dq.tensor(*([dq.ry(np.pi/2)] * N)).asdense()

Jzz = dq.zeros_like(rho).asdense()

for i in range(N):
    for j in range(i):
        Jzz += (sz_list[i] @ sz_list[j]) * J0[i,j]

times = np.linspace(0, tau, 20)

Sx_list = [dq.tensor(x) for x in sx_list]

Sx = functools.reduce(lambda x,y: x+y, Sx_list).to_jax() * 0.5

def evo(rho):
    result = dq.mesolve(Jzz, c_ops, rho, times, solver=dq.solver.Dopri8(rtol=1e-12, atol=1e-12))
    return result.states[-1]

for i in range(Nperiod):
    rho = RxT @ rho @ Rx
    rho = evo(rho)
    rho = Rx @ rho @ RxT
    rho = RyT @ rho @ Ry
    rho = evo(rho)
    rho = Ry @ rho @ RyT
    rho = evo(rho)

# highest qfi obtained after an extra rotation along x
rho = RxT @ rho @ Rx
rho = evo(rho)
rho = Rx @ rho @ RxT

# calculate collective spin operator Sz
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

# parity oscillation
rho = Ry.dot(rho.dot(RyT))
spin_up_count = np.bitwise_count(np.arange(0, 1<<N, dtype=int))

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

# Fig 5 inset
np.savetxt('./result/parity_local_dephasing.csv', np.stack((theta, parity), axis=1))
# np.savetxt('./result/parity_global_dephasing.csv', np.stack((theta, parity), axis=1))