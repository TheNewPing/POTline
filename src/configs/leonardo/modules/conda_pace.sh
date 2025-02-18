#!/bin/bash

module load cudnn/8.9.7.29-12--gcc--12.2.0-cuda-12.1

source $(conda info --base)/etc/profile.d/conda.sh
conda activate pl
