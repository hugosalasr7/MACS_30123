import pset1_30123
import numpy as np
import scipy.stats as sts
from numba.pycc import CC
from mpi4py import MPI
import matplotlib.pyplot as plt
import time

def find_opt_rho():

  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()
  size = comm.Get_size()

  start = time.time()

  mu = 3.0 # Mean
  sigma = 1.0 # Std
  S = int(1000) # Set the number of lives to simulate
  T = int(4160) # Set the number of periods for each simulation

  #If first core, create epsilon matrix. Then broadcast to the rest
  if rank == 0:
    np.random.seed(0)
    eps_mat = sts.norm.rvs(loc=0, scale=sigma, size=(T, S))
  else:
    eps_mat = np.empty([T, S], dtype ='float')

  comm.Bcast(eps_mat, root = 0) #Broadcast from 0 to all other cores

  z_mat = np.zeros((T, S))
  z_0 = mu
  lv, hv = -0.95, 0.95
  num_rhos = 200
  grid_rho = np.linspace(lv, hv, num=num_rhos) #grid of all rhos
  # Grid for rhos relevant for this core
  rhos_per_core = np.arange(0, num_rhos + num_rhos/size, num_rhos/size)
  grid_rho_ccore = grid_rho[int(rhos_per_core[rank]): int(rhos_per_core[rank+1])]

  avg_days_b4_sick = pset1_30123.sim_loop_jit_vrho(z_0, S, T, eps_mat, 
                                                   z_mat, mu, sigma, 
                                                   grid_rho_ccore)

  rhos_sick = np.empty([int(num_rhos/size), 2], dtype ='float')
  rhos_sick[:,1] = avg_days_b4_sick
  rhos_sick[:,0] = grid_rho_ccore

  #print(f'Core {rank}, rhos: ({rhos_per_core[rank]}, {rhos_per_core[rank+1]}), avg_sick: {rhos_sick.shape}')

  rhos_sick_all = None
  if rank == 0:
    rhos_sick_all = np.empty([num_rhos, 2], dtype ='float')

  comm.Gather(sendbuf=rhos_sick, recvbuf=rhos_sick_all, root=0)

  if rank == 0:
    fig = plt.figure()
    ax = plt.axes()
    ax.plot(rhos_sick_all[:,0], rhos_sick_all[:,1])
    plt.xlabel('Rho value')
    plt.title('Avg number of periods that it takes to negative health')
    plt.savefig("seconds_per_core.png")

    max_sick = max(rhos_sick_all[:,1])
    opt_rho = rhos_sick_all[rhos_sick_all[:,1]==max_sick, 0][0]
    print(f'Optimal rho: {opt_rho}, average persistence associated: {max_sick}')

    end = time.time()
    print("Elapsed seconds = %s" % (end - start))

if __name__ == '__main__':
  find_opt_rho()