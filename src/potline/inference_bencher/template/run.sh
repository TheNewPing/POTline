#!/bin/bash

# Collect arguments
n_cpu=$1
lammps_bin_path=$2
bench_potential_in_path=$3
prerun_steps=$4
max_steps=$5
out_path=$6


# Change to the output directory
cd ${out_path}

export OMP_NUM_THREADS=${n_cpu}
cp ${bench_potential_in_path} ${out_path}

echo "start"
start_time=`date +%s`
mpirun --oversubscribe -np ${n_cpu} ${lammps_bin_path} -in "${out_path}/bench.in" -v steps ${prerun_steps}
echo "prerun done"
mid_time=`date +%s`
mpirun --oversubscribe -np ${n_cpu} ${lammps_bin_path} -in "${out_path}/bench.in" -v steps ${max_steps}
end_time=`date +%s`
echo "finished"

runtime1=$((mid_time-start_time))
runtime2=$((end_time-mid_time))
runtime3=$((runtime2-runtime1))

# Write timings to a file
timings_file="${out_path}/bench_timings.csv"
echo "start_time,mid_time,end_time,prerun_steps,max_steps,time_diff" > $timings_file
echo "$start_time,$mid_time,$end_time,${prerun_steps},${max_steps},$runtime3" >> $timings_file

exit 0
