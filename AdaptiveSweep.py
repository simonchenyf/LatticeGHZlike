"""
File: AdaptiveSweep.py
Author: Yaofeng Chen, Xuanchen Zhang
Organization: Tsinghua University

Description:

    Used to obtain results for higher F_Q/F_Q^eff values based on lower F_Q/F_Q^eff results.
    One should first use 'AdaptiveBegin.py' to generate results for the lower F_Q/F_Q^eff value.

    This code produces data for Fig. 6, corresponding to F_Q/F_Q^eff = 0.7 (also called ratio in Appendix E),
    and Figs. 4 (a2) and (b2) in the main text, corresponding to F_Q/F_Q^eff = 0.8.

    Data for Figs. 4 (a3) and (b3) are also obtained using this code.

Make sure you run the following codes before running this code:
    AdaptiveBegin.py
    
Next file to use: ExportTime.py

License:
MIT License

"""

import numpy as np
from util import GHZlike
import os
import multiprocessing

def sweep_alpha(args):

    alpha_list, test, index, pre_name = args

    tau_worker = np.load(f'./data/{pre_name}/Worker_{index}.npz')

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

            tau0 = tau_worker['tau'][i,j]

            if tau >= tau0:
                tau = tau0
                qfi0 = tau_worker['qfi'][i,j]

            else:
                if not os.path.isdir(test.dir_path + f'/alpha={np.round(alpha,2)}/L={L}/tau={np.round(tau,6)}'):
                    os.mkdir(test.dir_path + f'/alpha={np.round(alpha,2)}/L={L}/tau={np.round(tau,6)}')
                test.set_tau(tau)
                test.run()
                qfi0 = np.max(test.qfi.cpu().numpy())
            
            if qfi0 >= test.ratio * np.max(test.FQ_eff):
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
                        if not os.path.isdir(test.dir_path + f'/alpha={np.round(alpha,2)}/L={L}/tau={np.round(tau,6)}'):
                            os.mkdir(test.dir_path + f'/alpha={np.round(alpha,2)}/L={L}/tau={np.round(tau,6)}')
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

    # ------------------------  modify the ratio values below ------------------------

    test_name = 'ratio7_1d' # 'ratio8_1d'
    # test_name = 'ratio7_2d' # 'ratio8_2d'

    pre_name = 'ratio6_1d' # 'ratio7_1d'
    # pre_name = 'ratio6_2d' # 'ratio7_2d'

    test = GHZlike(Nbatch=1000, ratio=0.7, test_name=test_name, device='cuda')

    # ------------------------  modify the ratio values above ------------------------

    alpha_lists = np.arange(0.1,6.05,0.1).reshape(10,6).T.tolist()

    if not os.path.isdir(test.dir_path):
        os.mkdir(test.dir_path)

    pool = multiprocessing.Pool()

    pool.map(sweep_alpha, zip(alpha_lists, [test]*len(alpha_lists), list(range(len(alpha_lists))), [pre_name]*len(alpha_lists)))

    pool.close()
    pool.join()
