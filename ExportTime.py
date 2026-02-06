"""
File: ExportTime.py
Author: Yaofeng Chen, Xuanchen Zhang
Organization: Tsinghua University

Description:

    This code extracts the data calculated by AdaptiveBegin.py and AdaptiveSweep.py, and
    store the tuple (L, tau, F_Q/F_Q^eff) for different alpha values (interaction range).

Make sure you run the following codes before running this code:
    AdaptiveBegin.py
    AdaptiveSweep.py (when applied)

Next file to use: FindScaling.py, FindTotalTimeScaling.py

License:
MIT License

"""

import numpy as np

test_name = 'ratio8_1d' # to extract data
scaling_name = 'ratio8_1d_scaling' # to save scaling results

alpha_lists = np.arange(0.1,6.05,0.1).reshape(10,6).T.tolist()

# ------------------------  switch for 1d and 2d below ------------------------

# for 1d
L_list = [5, 8, 13, 20, 32 ,50]

# # for 2d
# L_list = [10, 13, 16, 20, 25, 32]

# ------------------------  switch for 1d and 2d above ------------------------


for index, alpha_worker in enumerate(alpha_lists):

    data = np.load(f'./data/{test_name}/Worker_{index}.npz')

    for i, alpha in enumerate(alpha_worker):

        tau_alpha = np.stack((L_list, data['tau'][i], data['ratio'][i]), axis=1)
        np.savetxt(f'./data/{scaling_name}/alpha={np.round(alpha,2)}.csv', tau_alpha, delimiter=',')