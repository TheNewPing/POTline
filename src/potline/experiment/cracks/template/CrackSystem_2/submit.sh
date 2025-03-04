#!/bin/bash

# Collect the input parameters
LMMP=$1
cpus_per_task=$2
ntasks=$3

export MKL_NUM_THREADS=${cpus_per_task}
export OMP_NUM_THREADS=${cpus_per_task}

results_path="../../coeff/lefm_coeffs/lefm_paras.CrackSystem_2"

# get the equilibrium constants
a0=$(grep 'a0 =' ${results_path} | awk '{print $3}')
mass=95.95

KI=$(grep '#K_I=' ./lefm_coeffs/lefm_paras.CrackSystem_2 | awk '{print $2}')
Kstart=`printf "%.0f" $(bc <<< "$KI*100-10")`
Kstop=`printf "%.0f" $(bc <<< "$Kstart+100")`

eval srun ${LMMP} -in in.cracksystem_2 -v a0 ${a0} -v m ${mass} -v CrkSys 2 -v Kstart ${Kstart} -v Kstop ${Kstop}
zip sim_result.zip dump*
rm dump*
