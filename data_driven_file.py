import numpy as np
import pandas as pd

from tools import Buffer_gen

path = ''

daily_feats = pd.read_excel(path + 'dataset/DailyFeats.xlsx')
Num_day = len(daily_feats)

Params = np.load(path + 'dataset/Params.npz')
Phis = Params['Phis']
Alphas = Params['Alphas']
Betas = Params['Betas']
RPs = Params['RPs']
Proportion = Params['Proportion']
Mus = Params['Mus']
Sigmas = Params['Sigmas']
Ps = Params['Ps']

dfs = []
dfts = []
for i in range(Num_day):
    pattern = daily_feats.iloc[i]['pattern']
    feat_1 = daily_feats.iloc[i]['feat_1']
    feat_2 = daily_feats.iloc[i]['feat_2']
    dft = pd.read_excel(path + 'dataset/ResDay_' + str(i) + '.xlsx')
    df = pd.read_excel(path + 'dataset/ResDay_' + str(i) + 'Plus.xlsx')
    dft['pattern'] = pattern
    df['pattern'] = pattern
    dft['feat_1'] = feat_1
    df['feat_1'] = feat_1
    dft['feat_2'] = feat_2
    df['feat_2'] = feat_2
    dfs.append(df)
    dfts.append(dft)

df_Buffer = Buffer_gen(dfs, daily_feats['pattern'])
