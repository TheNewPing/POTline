#!/bin/bash

module load spack/0.21.0-68a
spack load miniconda3

module load cudnn/8.9.7.29-12--gcc--12.2.0-cuda-12.1

source $(conda info --base)/etc/profile.d/conda.sh
conda activate mace
