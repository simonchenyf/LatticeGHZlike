"""
File: FindTotalTimeScaling.py
Author: Yaofeng Chen, Xuanchen Zhang
Organization: Tsinghua University

Description:

    This code plots Fig. 3e and Fig. 3f in the main text.

Make sure you run the following codes before running this code:
    AdaptiveBegin.py
    AdaptiveSweep.py (when applied)
    ExportTime.py

License:
MIT License

"""

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

test_name = 'ratio8_1d'
scaling_name = 'ratio8_1d_scaling'

alpha_list = np.arange(0.1, 6.05, 0.1)

special_T = []

for alpha in alpha_list:
    scaling_data = np.loadtxt(f'{scaling_name}/alpha={np.round(alpha,2)}.csv', delimiter=',')
    L_list = scaling_data[:,0]
    tau_list = scaling_data[:,1]

    total_time = np.zeros_like(L_list)

    for i, L in enumerate(L_list):
        tau = tau_list[i]
        data = np.load(f'{test_name}/alpha={np.round(alpha,2)}/L={int(L)}/tau={np.round(tau,6)}/dtwa.npz', allow_pickle=True)['arr_0'].item()
        total_time[i] = data['tau'] * np.argmax(data['qfi'])

    np.savetxt(f'{test_name}/alpha={np.round(alpha,2)}/total_time.csv', np.stack((L_list, total_time), axis=1), delimiter=',')
    plt.plot(L_list, total_time)
    plt.xlabel('L')
    plt.xscale('log')
    plt.ylabel('T')
    plt.yscale('log')
    plt.title(f'alpha={np.round(alpha,2)}')
    plt.savefig(f'{test_name}/alpha={np.round(alpha,2)}/total_time.png')
    plt.close()

    if np.isclose(alpha, 1.0) or np.isclose(alpha, 2.0) or np.isclose(alpha, 3.0) or np.isclose(alpha, 6.0):
        special_T.append(total_time)
        print(alpha, len(special_T))

d = 1

special_T = np.array(special_T)

logL = np.log(L_list)
logT = np.log(special_T)

plt.scatter(logL, logT[0], label='alpha=1.0')
hopt1, _ = curve_fit(lambda x, h: -(d-1)*x + np.log(x) + h, logL, logT[0])
plt.plot(logL, -(d-1) * logL + np.log(logL) + hopt1, label='fit1')
print(hopt1, logT[0], -(d-1) * logL[-1] + np.log(logL[-1]) + hopt1)

plt.scatter(logL, logT[1], label='alpha=2.0')
hopt2, _ = curve_fit(lambda x, h: -(d-2)*x + np.log(x) + h, logL, logT[1])
plt.plot(logL, -(d-2) * logL + np.log(logL) + hopt2, label='fit2')
print(hopt2, logT[1], -(d-2) * logL[-1] + np.log(logL[-1]) + hopt2)

plt.scatter(logL, logT[2], label='alpha=3.0')
hopt3, _ = curve_fit(lambda x, h: -(d-3)*x + np.log(x) + h, logL, logT[2])
plt.plot(logL, -(d-3) * logL + np.log(logL) + hopt3, label='fit3')
print(hopt3, logT[2], -(d-3) * logL[-1] + np.log(logL[-1]) + hopt3)

plt.scatter(logL, logT[3], label='alpha=6.0')
hopt4, _ = curve_fit(lambda x, h: 2*x + np.log(x) + h, logL, logT[3])
plt.plot(logL, 2 * logL + np.log(logL) + hopt4, label='fit4')
print(hopt4, logT[3], 2 * logL[-1] + np.log(logL[-1]) + hopt4)


# plt.legend()
plt.savefig(f'./result/{test_name}/TotalTime.png')
plt.close()

np.savez(f'./result/{test_name}/TotalTimeFitting.npz', alpha=np.array([1.0, 2.0, 3.0, 6.0]), h=np.array([hopt1, hopt2, hopt3, hopt4]))
np.savetxt(f'./result/{test_name}/TotalTimeFitting.csv', np.stack((np.array([1.0, 2.0, 3.0, 6.0]), np.array([hopt1, hopt2, hopt3, hopt4]).reshape(-1)), axis=1), delimiter=',')