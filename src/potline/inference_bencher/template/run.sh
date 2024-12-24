#!/bin/bash

# Collect arguments
n_cpu=$1
lammps_bin_path=$2
prerun_steps=$3
max_steps=$4

export MKL_NUM_THREADS=${n_cpu}
export OMP_NUM_THREADS=${n_cpu}

echo "start"
start_time=`date +%s`
eval mpirun -np 1 --bind-to core ${lammps_bin_path} -in "bench.in" -v steps ${prerun_steps}
echo "prerun done"
mid_time=`date +%s`
eval mpirun -np 1 --bind-to core ${lammps_bin_path} -in "bench.in" -v steps ${max_steps}
end_time=`date +%s`
echo "finished"

runtime1=$((mid_time-start_time))
runtime2=$((end_time-mid_time))
runtime3=$((runtime2-runtime1))

# Write timings to a file
timings_file="bench_timings.csv"
echo "start_time,mid_time,end_time,prerun_steps,max_steps,n_cpu,time_diff" > $timings_file
echo "$start_time,$mid_time,$end_time,${prerun_steps},${max_steps},${n_cpu},$runtime3" >> $timings_file

exit 0
