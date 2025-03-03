#!/bin/bash

# Collect arguments
lammps_bin_path=$1
prerun_steps=$2
max_steps=$3
cpus_per_task=$4
ntasks=$5

export MKL_NUM_THREADS=${cpus_per_task}
export OMP_NUM_THREADS=${cpus_per_task}

echo "start"
start_time=`date +%s`
eval srun -n ${ntasks} ${lammps_bin_path} -in "bench.in" -v steps ${prerun_steps}
echo "prerun done"
mid_time=`date +%s`
eval srun -n ${ntasks} ${lammps_bin_path} -in "bench.in" -v steps ${max_steps}
end_time=`date +%s`
echo "finished"

runtime1=$((mid_time-start_time))
runtime2=$((end_time-mid_time))
runtime3=$((runtime2-runtime1))

# Write timings to a file
timings_file="bench_timings.csv"
echo "start_time,mid_time,end_time,prerun_steps,max_steps,cpus_per_task,ntasks,time_diff" > $timings_file
echo "$start_time,$mid_time,$end_time,${prerun_steps},${max_steps},${cpus_per_task},${ntasks},$runtime3" >> $timings_file

exit 0
