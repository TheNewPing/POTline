#!/bin/bash

module load 2022
module load cuDNN/8.4.1.50-CUDA-11.7.0
export LD_LIBRARY_PATH=/home/erodaro/.conda/envs/pl/lib/:$LD_LIBRARY_PATH

source $(conda info --base)/etc/profile.d/conda.sh
conda activate pl
