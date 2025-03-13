#!/bin/bash
cpus_per_task=$1
ntasks=$2

export MKL_NUM_THREADS=${cpus_per_task}
export OMP_NUM_THREADS=${cpus_per_task}

# first, solve the coeffs. 
python Solve_aniso_coeff.py --resultspath ../../../properties_bench/${SLURM_ARRAY_TASK_ID}/data/results.txt
