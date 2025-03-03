#!/bin/bash

# Collect the input parameters
LMMP=$1
db_path=$2
cpus_per_task=$3
ntasks=$4

export MKL_NUM_THREADS=${cpus_per_task}
export OMP_NUM_THREADS=${cpus_per_task}

rm energy.dat

for((i=0;i<=306;i=i+1))
do
lmpdata=${db_path}/lmp.screw_DB_$i

#sed -i 's/2 atom types/1 atom types/' $lmpdata
eval srun -n ${ntasks} ${LMMP} -in lmp.in -v lmpdata $lmpdata

echo $i ' is done!'
done
