#!/bin/bash

# Collect the input parameters
LMMP=$1
cpus_per_task=$2
ntasks=$3

export MKL_NUM_THREADS=${cpus_per_task}
export OMP_NUM_THREADS=${cpus_per_task}

results_path=../../../properties_bench/${SLURM_ARRAY_TASK_ID}/data/results.txt
coeff_path=../../coeff/${SLURM_ARRAY_TASK_ID}/lefm_coeffs/lefm_paras.CrackSystem_3

# get the equilibrium constants
a0=$(grep 'a0 =' ${results_path} | awk '{print $3}')
mass=95.95

KI=$(grep '#K_I=' ${coeff_path} | awk '{print $2}')
Kstart=`printf "%.0f" $(bc <<< "$KI*100-10")`
Kstop=`printf "%.0f" $(bc <<< "$Kstart+100")`

cp ${coeff_path} .
sed -i '/Fe$/ s/$/ Fe/' ./potential.in

eval srun -N 1 -n ${ntasks} ${LMMP} -in in.cracksystem_3 -v a0 ${a0} -v m ${mass} -v CrkSys 3 -v Kstart ${Kstart} -v Kstop ${Kstop}
zip cs3_result.zip *dump*
rm *dump*
