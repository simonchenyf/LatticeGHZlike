"""
File: HeatMap.py
Author: Yaofeng Chen, Xuanchen Zhang
Organization: Tsinghua University

Description:

    This code calculates the data for Figs. 4 (a1) and (a2), i.e. different F_Q/F_Q^eff
    values under different alpha (interaction range) and tau (pulse separation).

License:
MIT License

"""

import numpy as np
from util import GHZlike
import os

test_name = 'heatmap_1d' # 'heatmap_2d'

alpha = 1.5

# ------------------------  switch for 1d and 2d below ------------------------

# for 1d
L_list = [10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50]

# # for 2d
# L_list = [4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]

# ------------------------  switch for 1d and 2d above ------------------------

Tau_range = 10 ** np.linspace(0, -2, 101)

test = GHZlike(Nbatch=1000, ratio=0, test_name=test_name, device='cuda')

if not os.path.isdir(test.dir_path):
    os.mkdir(test.dir_path)

with open(test.dir_path + '/info.txt', 'a') as f:
    f.write(
        f'''
        ------ test: {test_name} ------ \n
            L: {L_list} \n
            Tau: {Tau_range.tolist()} \n
            Nbatch: {test.Nbatch} \n
        '''
)

test.set_alpha(alpha)

ratio = np.zeros((len(L_list), len(Tau_range)))

if not os.path.isdir(test.dir_path + f'/alpha=1.5'):
    os.mkdir(test.dir_path + f'/alpha=1.5')

for j, L in enumerate(L_list):
    if not os.path.isdir(test.dir_path + f'/alpha=1.5/L={L}'):
        os.mkdir(test.dir_path + f'/alpha=1.5/L={L}')
    test.set_config(L, 1)
    test.effective()

    for i, tau in enumerate(Tau_range):
        if not os.path.isdir(test.dir_path + f'/alpha=1.5/L={L}/tau={np.round(tau,6)}'):
            os.mkdir(test.dir_path + f'/alpha=1.5/L={L}/tau={np.round(tau,6)}')
        test.set_tau(tau)
        test.run()
    
        ratio[j,i] = np.max(test.qfi.cpu().numpy()) / np.max(test.FQ_eff)


coord = np.meshgrid(Tau_range, np.array(L_list))
data = np.stack((coord[1].reshape(-1), coord[0].reshape(-1), ratio.reshape(-1)), axis=-1)

# fig 4 a1 and b1
np.savetxt(f'./result/heatmap1d.csv', data, delimiter=',')
# np.savetxt(f'./result/heatmap2d.csv', data, delimiter=',')