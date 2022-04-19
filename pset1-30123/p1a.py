import numpy as np
import scipy.stats as sts
from numba import jit
import time
from numba.pycc import CC

def sim_loop(z_0, S, T, eps_mat, z_mat, rho, mu):
    for s_ind in range(S):
        z_tm1 = z_0
        for t_ind in range(T):
            e_t = eps_mat[t_ind, s_ind]
            z_t = rho * z_tm1 + (1 - rho) * mu + e_t
            z_mat[t_ind, s_ind] = z_t
            z_tm1 = z_t

cc = CC('pset1_30123')
@cc.export('sim_loop_jit', 'f4, i4, i4, f8[:,:], f8[:,:], f4, f4')

def sim_loop_jit(z_0, S, T, eps_mat, z_mat, rho, mu):
    for s_ind in range(S):
        z_tm1 = z_0
        for t_ind in range(T):
            e_t = eps_mat[t_ind, s_ind]
            z_t = rho * z_tm1 + (1 - rho) * mu + e_t
            z_mat[t_ind, s_ind] = z_t
            z_tm1 = z_t

cc.compile()

import pset1_30123

def simulate_health(rho = 0.5, mu = 3.0, sigma = 1.0, numba = False):

    # Set model parameters
    rho = rho
    mu = mu
    sigma = sigma
    z_0 = mu

    # Set simulation parameters, draw all idiosyncratic random shocks,
    # and create empty containers
    S = 1000 # Set the number of lives to simulate
    T = int(4160) # Set the number of periods for each simulation
    np.random.seed(25)
    eps_mat = sts.norm.rvs(loc=0, scale=sigma, size=(T, S))
    z_mat = np.zeros((T, S))

    if numba:
        pset1_30123.sim_loop_jit(z_0, S, T, eps_mat, z_mat, rho, mu)
    else:
        sim_loop(z_0, S, T, eps_mat, z_mat, rho, mu)


#Without Numba
start = time.time()
simulate_health(numba = False)
end = time.time()
print("Without Numba: elapsed seconds = %s" % (end - start))

#Without Numba
start = time.time()
simulate_health(numba = True)
end = time.time()
print("Numba: elapsed seconds = %s" % (end - start))