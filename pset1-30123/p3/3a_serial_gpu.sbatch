#!/bin/bash
#SBATCH --job-name=gpu      # job name
#SBATCH --output=gpu.out    # output log file
#SBATCH --error=gpu.err     # error file
#SBATCH --time=00:01:00     # 1 minute of wall time
#SBATCH --nodes=1           # 1 GPU node
#SBATCH --partition=gpu2    # GPU2 partition
#SBATCH --ntasks=1          # 1 CPU core to drive GPU
#SBATCH --gres=gpu:1        # Request 1 GPU
#SBATCH --account=macs30123 # specify the account

module load cuda
module load python/anaconda-2019.03

python3 ./raster_serial.py >> ./serial_3a.out

wait 

python3 ./raster_gpu.py >> ./gpu_3a.out