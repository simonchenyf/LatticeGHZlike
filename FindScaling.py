"""
File: FindScaling.py
Author: Yaofeng Chen, Xuanchen Zhang
Organization: Tsinghua University

Description:

    This code plots Figs. 4 (a2) and (b2) in the main text, and Fig. 6 in Appendix E,
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

# numerical dots for Figs 4 a2 and b2 and Fig 6
# the predicted solid lines are from eq (22)
np.savetxt(f'./result/{test_name}/scaling.csv', np.stack((alpha, -slope), axis=1), delimiter=',')