import numpy as np


class Simulation_Env:
    def __init__(self, Phis, Alphas, Betas, Mus, Sigmas, Ps, RPs, Capacity, MaxT):
        self.Phi = None
        self.Alpha = None
        self.Beta = None
        Num_N, Num_B = np.shape(Alphas)
        self.Num_H = Num_N
        self.Num_B = Num_B
        self.Phis = Phis
        self.Alphas = Alphas
        self.Betas = Betas
        self.Mus = Mus
        self.Sigmas = Sigmas
        self.Ps = Ps
        self.RPs = RPs
        self.Capacity = Capacity
        self.MaxT = MaxT
        self.state = dict()

    def Arrive(self):
        tau = self.state['time']
        Y = np.random.exponential(1)
        coef = np.array([-0.5 * self.Phi[1], self.Phi[0] + self.Phi[1] * tau, -Y])
        res_val = np.roots(coef)
        ti = np.max(res_val)
        self.state['time'] = tau - ti
        return tau - ti

    def Choice(self, Prices, Mode='MNL'):
        capacity = self.state['capacity']
        real_Prices = []
        for ib in range(self.Num_B):
            if capacity[ib] <= 0:
                price_b = 1e6
            else:
                price_b = Prices[ib]
            real_Prices.append(price_b)
        real_Prices = np.array(real_Prices)
        VoBC = self.Alpha + self.Beta * real_Prices
        if Mode == 'MNP':
            UoBC = np.hstack([0, VoBC]) + np.random.normal(size=self.Num_B + 1)
        else:
            UoBC = np.hstack([0, VoBC]) + np.random.gumbel(size=self.Num_B + 1)
        c_id = np.argmax(UoBC)
        if c_id > 0:
            capacity[c_id - 1] -= 1
        self.state['capacity'] = capacity
        return c_id, real_Prices

    def update(self, Prices, Mode):
        time = self.Arrive()
        c_id, real_Prices = self.Choice(Prices, Mode)
        done = time <= 0
        state = self.state.copy()
        return state, time, c_id, real_Prices, done

    def reset(self, pattern):
        self.state['time'] = self.MaxT
        self.state['capacity'] = self.Capacity.copy()
        self.Phi = self.Phis[pattern - 1]
        self.Alpha = self.Alphas[pattern - 1]
        self.Beta = self.Betas[pattern - 1]
        return self.state

