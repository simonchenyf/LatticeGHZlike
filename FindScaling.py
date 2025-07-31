"""
File: FindScaling.py
Author: Yaofeng Chen, Xuanchen Zhang
Organization: Tsinghua University

Description:

    This code plots Fig. 3c, and Fig. 3d in the main text, and Fig. S2 in the Supplementary Information (SI),
    based on the data calculated by AdaptiveBegin.py and AdaptiveSweep.py.

Make sure you run the following codes before running this code:
    AdaptiveBegin.py
    AdaptiveSweep.py (when applied)
    ExportTime.py

License:
MIT License

"""

import numpy as np
import scipy
import matplotlib.pyplot as plt

test_name = 'ratio8_1d'
scaling_name = 'ratio8_1d_scaling'

alpha = np.arange(0.1,6.05,0.1)
slope = np.zeros_like(alpha)

for i, a in enumerate(alpha):
    data = np.loadtxt(f"./data/{scaling_name}/alpha={np.round(a,2)}.csv", delimiter=',')
    log_N = np.log10(data[:,0])
    log_tau = np.log10(data[:,1])
    fit = scipy.stats.linregress(log_N, log_tau)
    slope[i] = fit.slope
    plt.plot(log_N, fit.slope * log_N + fit.intercept)
    plt.scatter(log_N, log_tau)
    plt.title(f"slope={np.round(fit.slope,6)}, intercept={np.round(fit.intercept,6)}")
    plt.savefig(f'./data/{scaling_name}/alpha={np.round(a,2)}.png')
    plt.close()

plt.scatter(alpha, -slope)
plt.plot(np.linspace(0,2,2), -np.linspace(-2,0,2), 'r')
plt.plot(np.linspace(2,4,2), -np.linspace(0,-2,2), 'r')
plt.plot(np.linspace(4,6,2), -np.array([-2.,-2.]), 'r')

plt.savefig(f"./result/{test_name}/0_scaling.png")
np.savez(f"./result/{test_name}/0_scaling.npz", alpha=alpha, slope=slope)
np.savetxt(f'./result/{test_name}/0_scaling.csv', np.stack((alpha, -slope), axis=1), delimiter=',')