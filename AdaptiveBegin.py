"""
File: AdaptiveBegin.py
Author: Yaofeng Chen, Xuanchen Zhang
Organization: Tsinghua University

Description:

    Used to generate results for a given F_Q/F_Q^eff value.

    This code produces data for Fig. 6, corresponding to F_Q/F_Q^eff = 0.6 (also called ratio in Appendix E).

    The name 'AdaptiveBegin' indicates that the parameter are swept with F_Q/F_Q^eff = 0.6 first, 
    the results of which are used as starting points for F_Q/F_Q^eff = 0.7 (Fig. 6), 
    and in turn, F_Q/F_Q^eff = 0.8 (Figs. 4 (a2) and (b2)).

Next file to use: AdaptiveSweep.py

License:
MIT License

"""

import numpy as np
from util import GHZlike
import os
import multiprocessing

def sweep_alpha(args):

    alpha_list, test, index = args

    if isinstance(alpha_list, int):
        alpha_list = [alpha_list]
    
    alpha_range = np.array(alpha_list, dtype=np.float64)

    dlntau = 0.02 # increment step for sweeping ln(tau) 

    # ------------------------  switch for 1d and 2d below ------------------------

    # for 1d
    L_list = [5, 8, 13, 20, 32 ,50]

    # # for 2d
    # L_list = [10, 13, 16, 20, 25, 32]

    # ------------------------  switch for 1d and 2d above ------------------------

    tau_array = np.zeros((len(alpha_range), len(L_list)), dtype=np.float64)
    max_qfi = np.zeros((len(alpha_range), len(L_list)), dtype=np.float64)
    max_qfi_eff = np.zeros((len(alpha_range), len(L_list)), dtype=np.float64)

    with open(test.dir_path + '/info.txt', 'a') as f:
        f.write(
            f'''
            ------ test: {test.test_name} ------ \n
                 L: {L_list} \n
                alpha: {alpha_range.tolist()} \n
                dlntau: {dlntau} \n
                Nbatch: {test.Nbatch} \n
                ratio: {test.ratio} \n
                Run on: {test.device} \n
            '''
        )

    for i, alpha in enumerate(alpha_list):
        if not os.path.isdir(test.dir_path + f'/alpha={np.round(alpha,2)}'):
            os.mkdir(test.dir_path + f'/alpha={np.round(alpha,2)}')
        test.set_alpha(alpha)

        # ------------------------  switch for 1d and 2d below ------------------------

        tau = 1.0 # starting point of sweep for 1d
        # tau = 0.1 # starting point of sweep for 2d

        # ------------------------  switch for 1d and 2d above ------------------------

        for j, L in enumerate(L_list):
            if not os.path.isdir(test.dir_path + f'/alpha={np.round(alpha,2)}/L={L}'):
                os.mkdir(test.dir_path + f'/alpha={np.round(alpha,2)}/L={L}')

            # ------------------------  switch for 1d and 2d below ------------------------

            test.set_config(L, 1) # 1d
            # test.set_config(L, L) # 2d

            # ------------------------  switch for 1d and 2d above ------------------------

            test.effective()

            if not os.path.isdir(test.dir_path + f'/alpha={np.round(alpha,2)}/L={L}/tau={np.round(tau,6)}'):
                os.mkdir(test.dir_path + f'/alpha={np.round(alpha,2)}/L={L}/tau={np.round(tau,6)}')
            test.set_tau(tau)
            test.run()

            if np.max(test.qfi.cpu().numpy()) >= test.ratio * np.max(test.FQ_eff):
                crit = True
                while crit:
                    tau = np.power(10, np.log10(tau) + dlntau)
                    if not os.path.isdir(test.dir_path + f'/alpha={np.round(alpha,2)}/L={L}/tau={np.round(tau,6)}'):
                        os.mkdir(test.dir_path + f'/alpha={np.round(alpha,2)}/L={L}/tau={np.round(tau,6)}')
                    test.set_tau(tau)
                    test.run()
                    if np.max(test.qfi.cpu().numpy()) < test.ratio * np.max(test.FQ_eff):
                        crit = False
                        tau = np.power(10, np.log10(tau) - dlntau)
                        test.set_tau(tau)
                        test.run()
            else:
                crit = True
                while crit:
                    tau = np.power(10, np.log10(tau) - dlntau)
                    if not os.path.isdir(test.dir_path + f'/alpha={np.round(alpha,2)}/L={L}/tau={np.round(tau,6)}'):
                        os.mkdir(test.dir_path + f'/alpha={np.round(alpha,2)}/L={L}/tau={np.round(tau,6)}')
                    test.set_tau(tau)
                    test.run()
                    if np.max(test.qfi.cpu().numpy()) >= test.ratio * np.max(test.FQ_eff):
                        crit = False

            tau_array[i,j] = tau
            max_qfi[i,j] = np.max(test.qfi.cpu().numpy())
            max_qfi_eff[i,j] = np.max(test.FQ_eff)

            np.savez(test.dir_path + f'/Worker_{index}.npz', tau=tau_array[:(i+1)], qfi=max_qfi[:(i+1)], qfi_eff=max_qfi_eff[:(i+1)], ratio=max_qfi[:(i+1)]/max_qfi_eff[:(i+1)])


if __name__ == '__main__':

    test_name = 'ratio6_1d'
    # test_name = 'ratio6_2d'
    test = GHZlike(Nbatch=1000, ratio=0.6, test_name=test_name, device='cuda')
    alpha_lists = np.arange(0.1,6.05,0.1).reshape(10,6).T.tolist()

    if not os.path.isdir(test.dir_path):
        os.mkdir(test.dir_path)

    pool = multiprocessing.Pool()

    pool.map(sweep_alpha, zip(alpha_lists, [test]*len(alpha_lists), list(range(len(alpha_lists)))))

    pool.close()
    pool.join()
