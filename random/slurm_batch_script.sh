#!/bin/bash
#SBATCH --job-name=pyt_gpu_test
#SBATCH --output=output_%j.txt
#SBATCH --error=error_%j.txt
#SBATCH --partition=gpu_a100
#SBATCH --gpus=a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00

module load 2023
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

srun python /home/erodaro/POTline/src/potline/dispatcher/slurm/template/pyt_gpu_test.py
