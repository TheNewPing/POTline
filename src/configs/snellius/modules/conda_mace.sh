#!/bin/bash

module load 2024
module load CUDA/12.6.0
export LD_LIBRARY_PATH=/home/erodaro/.conda/envs/mace/lib/:$LD_LIBRARY_PATH

source $(conda info --base)/etc/profile.d/conda.sh
conda activate mace
