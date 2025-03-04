#!/bin/bash
cpus_per_task=$3
ntasks=$4

export MKL_NUM_THREADS=${cpus_per_task}
export OMP_NUM_THREADS=${cpus_per_task}

# first, solve the coeffs. 
python Solve_aniso_coeff.py --resultspath ../../../properties_bench/${SLURM_ARRAY_TASK_ID}/data/results.txt
