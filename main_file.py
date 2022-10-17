import numpy as np
import pandas as pd

from Simulation import Simulation_Env
from tools import RP_solver, Data_generate

path = 'D:/Dropbox/RM_P1/Synthetic_Experiment/'

Num_B = 3
Num_N = 4
Max_T = 30
Capacity = np.array([6, 9, 12])

Phis = np.array([[2.25, -0.1], [2.5, -0.075], [2.75, -0.05], [3, -0.025]])
Alphas = np.array([[-1.75, -1.25, -0.75], [-1.25, -1.05, -0.65],
                   [-1.05, -0.95, -0.55], [-0.95, -0.75, -0.35]])
Betas = np.array([[-0.0175, -0.02, -0.025], 
                  [-0.015, -0.0175, -0.02],
                  [-0.0125, -0.015, -0.0175], 
                  [-0.0075, -0.01, -0.0125]])
RPs = []
for i in range(Num_N):
    RP_val = RP_solver(Alphas[i], Betas[i])
    RPs.append(RP_val)
RPs = np.vstack(RPs)
Proportion = np.array([0.35, 0.4, 0.2, 0.05])
Mus = np.array([0.2, 0.3, 0.5, 0.8])
Sigmas = np.array([2, 1, 0.5, 0.1])
Ps = np.array([0.2, 0.3, 0.5, 0.8])
np.savez(path + 'dataset/Params_1.npz', Phis=Phis, Alphas=Alphas, Betas=Betas, RPs=RPs, Proportion=Proportion,
         Mus=Mus, Sigmas=Sigmas, Ps=Ps)

env0 = Simulation_Env(Phis, Alphas, Betas, Mus, Sigmas, Ps, RPs, Capacity, Max_T)

Num_day = 240
dfs, dfts = Data_generate(env0, Num_day, Proportion)

# %%
daily_feats = []
for i in range(240):
    daily_feats.append(dfs[i][:-1])
    dfts[i][3].to_excel(path + 'dataset/ResDay_' + str(i + 1) + '.xlsx', index=False)
    dfs[i][3].to_excel(path + 'dataset/ResDay_' + str(i + 1) + 'Plus.xlsx', index=False)

daily_feats = pd.DataFrame(daily_feats, columns=['pattern', 'feat_1', 'feat_2'])
daily_feats.to_excel(path + 'dataset/DailyFeats_1.xlsx', index=False)
