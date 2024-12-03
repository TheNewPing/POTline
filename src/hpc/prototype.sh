#!/bin/bash

#SBATCH --time=8:00:00
#SBATCH --partition=gpu
#SBATCH -N 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 16
#SBATCH --mem=60G
#SBATCH --gpus=1

module purge

module load 2024
module load Miniconda3/24.7.1-0

module load 2022
module load cuDNN/8.4.1.50-CUDA-11.7.0
export LD_LIBRARY_PATH=/home/erodaro/.conda/envs/pl/lib/:$LD_LIBRARY_PATH

# Initialize Conda
source $(conda info --base)/etc/profile.d/conda.sh
conda activate pl

python src/hpc/tf_gpu_test.py
python src/main.py --config src/data/prototype.hjson --nohyper --fitted quick_config/pace
