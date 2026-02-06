"""
File: SweepDephasing.py
Author: Xuanchen Zhang, Yaofeng Chen
Organization: Tsinghua University

Description:

    This code obtains the quantum Fisher information for different dephasing strengh Gamma.
    
    This code produces Fig. 5 in the main text.

License:
MIT License

"""

import dynamiqs as dq
import numpy as np
import jax
import jax.numpy as jnp
import functools

dq.set_device('gpu', index=1)
# dq.set_device('cpu')
dq.set_precision('double')

L = 12
W = 1
N = L * W

state_list = [dq.fock_dm(2, 0)] * N
rho = dq.tensor(*state_list).asdense()

tau = 0.1/560
Nperiod = 15

alpha = 1.

J0 = 560 * (np.ones((N, N), dtype=np.float64) - np.eye(N, dtype=np.float64))   
for i in range(N):
    for j in range(N):
        if i != j:
            dist = np.sqrt((i // W - j // W) ** 2 + (i % W - j % W) ** 2)
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

Gamma_list = np.logspace(-1, 2, 31)

def evo(rho, c_ops):
    result = dq.mesolve(Jzz, c_ops, rho, times, solver=dq.solver.Dopri8(rtol=1e-12, atol=1e-12))
    return result.states[-1]

@jax.jit
def get_pred(rho):
    p, v = jnp.linalg.eigh(rho)

    Sx_coeff = jnp.abs(v.T.conj().dot(Sx.dot(v))) ** 2

    pdiff = p.reshape(-1,1) - p.reshape(1, -1)
    psum = p.reshape(-1,1) + p.reshape(1, -1)

    zero_index = (psum < 1e-10).astype(jnp.float64)

    pdiff = pdiff.at[:].multiply(1-zero_index)

    zero_index = zero_index.at[:].multiply(1e-10)

    psum = psum.at[:].add(zero_index)

    F = jnp.trace((pdiff**2 / psum).dot(Sx_coeff)) * 2

    return F, (F, jnp.max(p), jnp.min(p), jnp.sum(p))

optFQList = np.zeros_like(Gamma_list, dtype=np.float64)
rho0 = rho

# ------------------------  switch for local and global dephasing below ------------------------

# local dephasing
c_ops_0 = [sz_list[i].asdense() for i in range(N)]

# # global dephasing
# Sz_total = sz_list[0]
# for i in range(N-1):
#     Sz_total += sz_list[i+1]
# c_ops_0 = [Sz_total.asdense()]

# ------------------------  switch for local and global dephasing above ------------------------

for j, Ga in enumerate(Gamma_list):

    rho = rho0
    sqrt_Gamma_z = np.sqrt(Ga)

    c_ops = [term * sqrt_Gamma_z/2 for term in c_ops_0]

    all_pred = np.zeros((3*Nperiod+2,), dtype=np.float64)
    all_pred[0], haha = get_pred(rho.to_jax())


    for i in range(Nperiod):
        rho = RxT @ rho @ Rx
        rho = evo(rho, c_ops)
        rho = Rx @ rho @ RxT
        all_pred[3*i+1], haha = get_pred(rho.to_jax())

        rho = RyT @ rho @ Ry
        rho = evo(rho, c_ops)
        rho = Ry @ rho @ RyT
        all_pred[3*i+2], haha = get_pred(rho.to_jax())

        rho = evo(rho, c_ops)
        all_pred[3*i+3], haha = get_pred(rho.to_jax())


    rho = RxT @ rho @ Rx
    rho = evo(rho, c_ops)
    rho = Rx @ rho @ RxT
    all_pred[-1], haha = get_pred(rho.to_jax())

    optFQList[j] = np.max(all_pred.real)

# Fig 5
np.savetxt(f'./result/FQList_local_dephasing.csv', np.stack((Gamma_list, optFQList), axis=1), delimiter=',')
# np.savetxt(f'./result/FQList_global_dephasing.csv', np.stack((Gamma_list, optFQList), axis=1), delimiter=',')
