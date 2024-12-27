#!/bin/bash

module load 2024
module load cuDNN/9.5.0.50-CUDA-12.6.0

source $(conda info --base)/etc/profile.d/conda.sh
conda activate grace
