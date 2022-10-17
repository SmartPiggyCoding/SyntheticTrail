import numpy as np
import pandas as pd

from numba import jit
from math import log
from scipy.optimize import minimize


@jit
def safe_log(x, ZERO_LOWER_BOUND=1e-6):
    if x > ZERO_LOWER_BOUND:
        return log(x)
    log_lower_bound = log(ZERO_LOWER_BOUND)
    a = 1 / (3 * ZERO_LOWER_BOUND * (3 * log_lower_bound * ZERO_LOWER_BOUND) ** 2)
    b = ZERO_LOWER_BOUND * (1 - 3 * log_lower_bound)
    return a * (x - b) ** 3


def RP_solver(Alpha, Beta):
    def Obj_fun(alpha, beta, Prices):
        VoBC = alpha + beta * Prices
        PoBC = np.exp(VoBC) / (1 + np.sum(np.exp(VoBC)))
        res = np.sum(Prices * PoBC)
        return res

    Num_B = len(Alpha)
    obj_fun = lambda Prices: -1 * Obj_fun(Alpha, Beta, Prices)
    Prices_ina = np.ones(Num_B)
    bnds = [(0, None)] * Num_B
    res_val = minimize(obj_fun, Prices_ina, bounds=bnds)
    RP = np.ceil(res_val.x)
    return RP


def Data_generate(env, N_df, proportion):
    dfs = []
    dfts = []
    for n in range(N_df):
        rec = []
        pattern = np.random.choice(np.arange(1, 5), p=proportion)
        feat_a = np.random.normal(loc=env.Mus[pattern - 1], scale=env.Sigmas[pattern - 1])
        feat_b = np.random.binomial(n=1, p=env.Ps[pattern - 1])
        RP = env.RPs[pattern - 1]
        env.reset(pattern)
        done = False
        while not done:
            Prices = RP * np.random.choice(np.arange(1, 0.5, -0.1), size=env.Num_B)
            state, time, c_id, real_Prices, done = env.update(Prices, Mode='MNL')
            if time < 0:
                break
            rec.append([time] + list(real_Prices) + [c_id] + list(state['capacity']))
        df = pd.DataFrame(rec, columns=['time'] + ['Price_' + str(b) for b in range(1, env.Num_B + 1)] + ['Choice'] +
                                       ['Capacity_' + str(b) for b in range(1, env.Num_B + 1)])
        dft = df[df.Choice != 0]
        dft.reset_index(drop=True, inplace=True)
        dfs.append([pattern, feat_a, feat_b, df])
        dfts.append([pattern, feat_a, feat_b, dft])
    return dfs, dfts


def Buffer_gen(dfs, pattern_lst):
    Buffer_rec = []
    for i in range(len(dfs)):
        pattern = pattern_lst[i]
        df_i = dfs[i].copy()
        df_i['dba'] = df_i['time'].apply(lambda x: int(x // 1))
        for j in range(len(df_i)):
            Buffer_tmp = {'state': None, 'action': None, 'reward': None, 'next_state': None, 'pattern': None,
                          'dba': None}
            dba = df_i.iloc[j]['dba']
            action = df_i.iloc[j][['Price_' + str(b) for b in range(1, 4)]].values.tolist()
            next_state = df_i.iloc[j][['Capacity_' + str(b) for b in range(1, 4)]].values.tolist()
            bc = df_i.iloc[j]['Choice']
            if bc == 0:
                reward = 0
                state = next_state.copy()
            else:
                reward = action[int(bc - 1)]
                state = next_state.copy()
                state[int(bc - 1)] += 1
            Buffer_tmp['state'] = state
            Buffer_tmp['action'] = action
            Buffer_tmp['reward'] = reward
            Buffer_tmp['next_state'] = next_state
            Buffer_tmp['pattern'] = pattern
            Buffer_tmp['dba'] = dba
            Buffer_rec.append(Buffer_tmp)
    df_Buffer = pd.DataFrame(Buffer_rec)
    return df_Buffer
