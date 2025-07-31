"""
File: Dephasing.py
Author: Yaofeng Chen, Xuanchen Zhang
Organization: Tsinghua University

Description:

    This code obtains the evolution of quantum Fisher information
    for different dephasing strengh Gamma.
    
    This code produces Fig. 4a in the main text.

Next file to use: DephasingParity.py

License:
MIT License

"""

import dynamiqs as dq
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import functools

dq.set_device('gpu', index=0)
dq.set_precision('double')

N = 12

state_list = [dq.fock_dm(2, 0)] * N
rho = dq.tensor(*state_list).asdense()

tau = 0.1/560
Nperiod = 15

# ------------------------  switch for Gamma below ------------------------

# # Gamma = 0
# sqrt_Gamma_z = 0

# # Gamma = Gamma_0 / 2
# T2 = 0.069 * 2
# sqrt_Gamma_z = np.sqrt(2/T2)

# Gamma = Gamma_0
T2 = 0.069
sqrt_Gamma_z = np.sqrt(2/T2)

# ------------------------  switch for Gamma above ------------------------

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

# collapse operators
c_ops = [sz_list[i].asdense() * sqrt_Gamma_z/2 for i in range(N)]

times = np.linspace(0, tau, 20)

Sx_list = [dq.tensor(x) for x in sx_list]

Sx = functools.reduce(lambda x,y: x+y, Sx_list).to_jax() * 0.5

def evo(rho):
    result = dq.mesolve(Jzz, c_ops, rho, times, solver=dq.solver.Dopri8(rtol=1e-12, atol=1e-12))
    return result.states[-1]

@jax.jit
def get_qfi(rho):
    p, v = jnp.linalg.eigh(rho)

    Sx_coeff = jnp.abs(v.T.conj().dot(Sx.dot(v))) ** 2

    pdiff = p.reshape(-1,1) - p.reshape(1, -1)
    psum = p.reshape(-1,1) + p.reshape(1, -1)

    zero_index = (psum < 1e-10).astype(jnp.float64)

    # pdiff *= (1-zero_index)
    pdiff = pdiff.at[:].multiply(1-zero_index)

    # zero_index *= 1e-10
    zero_index = zero_index.at[:].multiply(1e-10)

    # psum += zero_index
    psum = psum.at[:].add(zero_index)

    F = jnp.trace((pdiff**2 / psum).dot(Sx_coeff)) * 2

    return F

all_pred = np.zeros((3*Nperiod+2,), dtype=np.float64)
all_pred[0] = get_qfi(rho.to_jax())

for i in range(Nperiod):
    rho = RxT @ rho @ Rx
    rho = evo(rho)
    rho = Rx @ rho @ RxT
    all_pred[3*i+1] = get_qfi(rho.to_jax())
    rho = RyT @ rho @ Ry
    rho = evo(rho)
    rho = Ry @ rho @ RyT
    all_pred[3*i+2] = get_qfi(rho.to_jax())
    rho = evo(rho)
    all_pred[3*i+3] = get_qfi(rho.to_jax())

rho = RxT @ rho @ Rx
rho = evo(rho)
rho = Rx @ rho @ RxT
all_pred[-1] = get_qfi(rho.to_jax())

np.save('./data/opensystem/qfi_dephasing.npy', all_pred)

np.save('./data/opensystem/endstate_dephasing.npy', rho.to_numpy())

t = np.arange(3*Nperiod+2) * tau

plt.plot(t, all_pred.real / N**2)
plt.savefig('./result/qfi_dephasing.png')
np.savetxt('./result/qfi_dephasing.csv', np.stack((t, all_pred.real), axis=1), delimiter=',')
