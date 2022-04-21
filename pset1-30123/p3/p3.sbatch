#!/bin/bash

#SBATCH --job-name=mpi_multi_job
#SBATCH --ntasks=10
#SBATCH --partition=broadwl
#SBATCH --constraint=fdr
#SBATCH --account=macs30123

# Load Python and MPI modules
module load python/anaconda-2019.03
module load intelmpi/2018.2.199+intel-18.0

# Run the python program with mpirun, using & to run jobs at the same time

mpirun -n 1 python3 ./numba_loop.py >> ./compiled.out 

# Wait until all simultaneous mpiruns are done
wait

mpirun -n 10 python3 ./sim_health_mult_rhos.py > ./p1b.out 