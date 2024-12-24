#!/bin/bash

module load cuDNN/8.9.2.26-CUDA-12.2.0

source $(conda info --base)/etc/profile.d/conda.sh
conda activate mace
