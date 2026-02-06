"""
File: FindTotalTimeScaling.py
Author: Yaofeng Chen, Xuanchen Zhang
Organization: Tsinghua University

Description:

    This code plots Figs. 4 (a3) and (b3) in the main text.

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

T_list = []

for alpha in alpha_list:
    scaling_data = np.loadtxt(f'{scaling_name}/alpha={np.round(alpha,2)}.csv', delimiter=',')
    L_list = scaling_data[:,0]
    tau_list = scaling_data[:,1]

    total_time = np.zeros_like(L_list)

    for i, L in enumerate(L_list):
        tau = tau_list[i]
        data = np.load(f'{test_name}/alpha={np.round(alpha,2)}/L={int(L)}/tau={np.round(tau,6)}/dtwa.npz', allow_pickle=True)['arr_0'].item()
        total_time[i] = data['tau'] * np.argmax(data['qfi'])

    # results here for alpha = 1. , 2. , 3. and 6. are used to produce Figs. 4 a3 and b3
    np.savetxt(f'{test_name}/alpha={np.round(alpha,2)}/total_time.csv', np.stack((L_list, total_time), axis=1), delimiter=',')

    if np.isclose(alpha, 1.0) or np.isclose(alpha, 2.0) or np.isclose(alpha, 3.0) or np.isclose(alpha, 6.0):
        T_list.append(total_time)

d = 1

T_list = np.array(T_list)

logL = np.log(L_list)
logT = np.log(T_list)

h_opt1, _ = curve_fit(lambda x, h: -(d-1)*x + np.log(x) + h, logL, logT[0])
h_opt2, _ = curve_fit(lambda x, h: -(d-2)*x + np.log(x) + h, logL, logT[1])
h_opt3, _ = curve_fit(lambda x, h: -(d-3)*x + np.log(x) + h, logL, logT[2])
h_opt4, _ = curve_fit(lambda x, h: 2*x + np.log(x) + h, logL, logT[3])

# the intercepts of the fitted solid lines in Figs. 4 a3 and b3
np.savetxt(f'./result/{test_name}/TotalTimeFitting.csv', np.stack((np.array([1.0, 2.0, 3.0, 6.0]), np.array([h_opt1, h_opt2, h_opt3, h_opt4]).reshape(-1)), axis=1), delimiter=',')