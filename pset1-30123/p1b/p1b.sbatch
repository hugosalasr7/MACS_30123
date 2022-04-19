#!/bin/bash

#SBATCH --job-name=mpi_multi_job
#SBATCH --ntasks=5
#SBATCH --partition=broadwl
#SBATCH --constraint=fdr
#SBATCH --account=macs30123

# Load Python and MPI modules
module load python/anaconda-2019.03
module load intelmpi/2018.2.199+intel-18.0

# Run the python program with mpirun, using & to run jobs at the same time

for i in {1..20}
do
    mpirun -n $i python3 ./p1b.py > ./p1b.out 
done

# Wait until all simultaneous mpiruns are done
wait