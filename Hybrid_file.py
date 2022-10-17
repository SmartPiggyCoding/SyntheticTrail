# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 17:05:32 2022

@author: jydong7
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from scipy.spatial import distance
from random import sample

# %% Data Generation Process

class PurchaseBehavior():
    
    def __init__(self, Alpha, Beta, featD, featC):
        self._NumB = len(Alpha)
        self._Alpha = Alpha
        self._Beta = Beta
        # Discrete features (Binary features)
        self._featD = featD
        self._NumD = len(featD)
        # Continuous features (Normal distributed features)
        self._featC = featC 
        self._NumC = len(featC)
    
    def ChoiceProb(self, Price):
        UoI = self._Alpha + self._Beta * Price
        PoI = np.exp(UoI) / (1 + np.sum(np.exp(UoI)))
        return PoI

    def CalculateRP(self):
        obj_fun = lambda X: -1 * np.sum(X * self.ChoiceProb(X))
        X_ina = np.ones(self._NumB)
        X_bnds = [(0, None)] * self._NumB
        res = minimize(obj_fun, X_ina, bounds=X_bnds, method='SLSQP')
        RP = np.ceil(res.x)
        return RP
    
    def step(self, Price):
        PoI = self.ChoiceProb(Price)
        P_val = np.hstack([1 - np.sum(PoI), PoI])
        c_res = np.random.choice(np.arange(self._NumB + 1), p=P_val)
        rev_res = c_res * Price[int(c_res - 1)]
        dis_feat = np.random.binomial(1, self._featD)
        con_feat = np.random.normal(self._featC)
        return rev_res, dis_feat, con_feat


alpha_1 = [0.1]
alpha_2 = [0.3]
beta_1 = [-0.05]
beta_2 = [-0.02]
featD_1 = [0.2]
featD_2 = [0.8]
featC_1 = [-4]
featC_2 = [4]
feats_1 = np.hstack([featD_1, featC_1])
feats_2 = np.hstack([featD_2, featC_2])

env1 = PurchaseBehavior(alpha_1, beta_1, featD_1, featC_1)
env2 = PurchaseBehavior(alpha_2, beta_2, featD_2, featC_2) 

RP_1 = env1.CalculateRP()
RP_2 = env2.CalculateRP()

Rec = []
Labels = []
for _ in range(200):
    if np.random.rand() < 0.7:
        price = np.random.uniform(RP_1 - 20, RP_1 + 20)
        rev, d_feats, c_feats = env1.step(price)
        Labels.append(1)
    else:
        price = np.random.uniform(RP_2 - 20, RP_2 + 20)
        rev, d_feats, c_feats = env2.step(price)
        Labels.append(2)
    Rec.append([price[0], rev, d_feats[0], c_feats[0]])

Prices = [item[0] for item in Rec]
Revs = [item[1] for item in Rec]

#%%

def TraindataGen(rec):
    train_data = []
    mean_prices = []
    mean_rev = []
    rec = pd.DataFrame(rec, columns=['price', 'rev', 'featD', 'featC'])
    rec['label'] = rec['price'] // 20
    for i in rec.label.unique():
        df_tmp = rec[rec.label == i]
        for _ in range(100):
            train_tmp = df_tmp.sample(n=30, replace=True)
            train_data.append(train_tmp.mean().values[:-1])
            mean_prices.append(train_tmp.mean(0)[0])
            mean_rev.append(train_tmp.mean(0)[1])
    return train_data, mean_prices, mean_rev


def est_fun(theta, feats, price):
    Num_M = len(feats) + 1
    theta_1 = theta[:Num_M]
    theta_2 = theta[Num_M:2*Num_M]
    a = theta_1 @ np.hstack([1, feats])
    b = theta_2 @ np.hstack([1, feats])
    return -1 * a**2 * price **2 + b * price


def reg_fun(train_data, theta):
    endog = []
    exog = []
    for item in train_data:
        price, rev, d_feats, c_feats = item
        feats = np.hstack([d_feats, c_feats])
        endog.append(rev)
        exog.append(est_fun(theta, feats, price))
    loss = distance.euclidean(np.array(endog), np.array(exog))
    return loss

train_data, mean_prices, mean_rev = TraindataGen(Rec)

obj_fun = lambda theta: reg_fun(train_data, theta)
theta_ina = np.random.randn(6) * 1e-2
res = minimize(obj_fun, theta_ina, bounds=[(-50, 50)]*6, method='SLSQP')
print(res)
est_theta = res.x



plt.figure(figsize=(8, 6))
x = np.linspace(0, 100)
y_1 = x * np.exp(alpha_1 + beta_1 * x) / (1 + np.exp(alpha_1 + beta_1 * x))
y_2 = x * np.exp(alpha_2 + beta_2 * x) / (1 + np.exp(alpha_2 + beta_2 * x))
hat_y_1 = est_fun(est_theta, feats_1, x)
hat_y_2 = est_fun(est_theta, feats_2, x)
plt.plot(x, y_1, 'r')
plt.plot(x, y_2, 'b')
plt.scatter(Prices, Revs, c=Labels)
# plt.scatter(mean_prices, mean_rev, c='k', marker='.')
plt.plot(x, hat_y_1, 'r+')
plt.plot(x, hat_y_2, 'b+')
plt.show()

print('The Regular Prices for DOA 1 and 2 are: ', RP_1, RP_2)
print('Estimated Optimal Prices for DOA 1: ', est_theta[3:6] @ np.hstack([1, feats_1]) / (2 * ((est_theta[:3] @ np.hstack([1, feats_1])) ** 2)))
print('Estimated Optimal Prices for DOA 2: ', est_theta[3:6] @ np.hstack([1, feats_2]) / (2 * ((est_theta[:3] @ np.hstack([1, feats_2])) ** 2)))