#!/bin/bash

export OMP_NUM_THREADS=${n_cpu}

echo "start"
start_time=`date +%s`
mpirun --oversubscribe -np ${n_cpu} ${lammps_bin_path} -in ${bench_potential_in_path} -v steps ${prerun_steps}
echo "prerun done"
mid_time=`date +%s`
mpirun --oversubscribe -np ${n_cpu} ${lammps_bin_path} -in ${bench_potential_in_path} -v steps ${max_steps}
end_time=`date +%s`
echo "finished"

runtime1=$((mid_time-start_time))
runtime2=$((end_time-mid_time))
runtime3=$((runtime2-runtime1))
steps_diff=$((max_steps-prerun_steps))

# Write timings to a file
timings_file="${out_path}/bench_timings.txt"
echo "start_time: $start_time" > $timings_file
echo "mid_time: $mid_time" >> $timings_file
echo "end_time: $end_time" >> $timings_file
echo "steps_diff: $steps_diff" >> $timings_file
echo "time_diff: $runtime3" >> $timings_file

exit 0
