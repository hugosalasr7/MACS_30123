import numpy as np
import scipy.stats as sts
from numba import jit
import time
from numba.pycc import CC
from mpi4py import MPI
import pset1_30123

def simulate_health(rho = 0.5, mu = 3.0, sigma = 1.0):

    start = time.time() #Start timing

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Set model parameters
    rho = rho
    mu = mu
    sigma = sigma
    z_0 = mu

    # Set simulation parameters, draw all idiosyncratic random shocks,
    # and create empty containers
    S = int(1000/size) # Set the number of lives to simulate
    T = int(4160) # Set the number of periods for each simulation
    np.random.seed(rank)
    eps_mat = sts.norm.rvs(loc=0, scale=sigma, size=(T, S))
    z_mat = np.zeros((T, S))

    pset1_30123.sim_loop_jit(z_0, S, T, eps_mat, z_mat, rho, mu)

    z_mat_all = None
    if rank == 0:
      z_mat_all = np.empty([T, S*size], dtype ='float')
    
    comm.Gather(sendbuf=z_mat, recvbuf=z_mat_all, root=0)

    if rank == 0:
      end = time.time() #End timing
      print(f'Elapsed seconds with {size} cores: {end - start}')


if __name__ == '__main__':
    simulate_health()