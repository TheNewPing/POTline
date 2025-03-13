#!/bin/bash

# Collect the input parameters
LMMP=$1
cpus_per_task=$2
ntasks=$3

export MKL_NUM_THREADS=${cpus_per_task}
export OMP_NUM_THREADS=${cpus_per_task}

eval srun -n ${ntasks} ${LMMP} -in input_BCC_init
zip screw_result.zip *dump*
rm *dump*
