#!/bin/bash

module load cuDNN/8.9.2.26-CUDA-12.2.0
export LD_LIBRARY_PATH=/home4/p319875/.conda/envs/mace/lib/:$LD_LIBRARY_PATH

source $(conda info --base)/etc/profile.d/conda.sh
conda activate mace
