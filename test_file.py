import numpy as np
from tools import RP_solver


def Arr_amt(Phi, Max_T):
    return Phi[0] * Max_T + 0.5 * Phi[1] * Max_T ** 2


def Choice_prob(Alpha, Beta, RP):
    VoBC = Alpha + Beta * RP
    PoBC = np.exp(VoBC) / (1 + np.sum(np.exp(VoBC)))
    return PoBC


def Test_params(Phi, Alpha, Beta, Max_T, Capacity, Prices):
    ArrAmt = Arr_amt(Phi, Max_T)
    pobc = Choice_prob(Alpha, Beta, Prices)
    occ_rate = (ArrAmt * pobc) / Capacity
    print('The expect number of customer arrival is: ', ArrAmt)
    print('The expect number of demand is: ', np.sum(pobc) * ArrAmt)
    print('The market share is: ', np.sum(pobc))
    print('The occupancy rate for each room type is: ', occ_rate)


phi = np.array([8, -0.02])
alpha = np.array([-0.95, -0.75, -0.35])
beta = -0.01
MaxT = 60
capacity = np.array([40, 50, 70])

RP_val = RP_solver(alpha, beta)
Test_params(phi, alpha, beta, MaxT, capacity, RP_val)
Test_params(phi, alpha, beta, MaxT, capacity, RP_val * 0.6)

