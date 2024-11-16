"""
This module is responsible for running LAMMPS benchmarks.
"""

import os
import subprocess
from string import Template

LAMMPS_BIN_PATH = ''
LAMMPS_IN_TEMPLATE_PATH = './template/bench.in'
BENCH_SCRIPT_TEMPLATE_PATH = './template/run.sh'

def run_benchmark(out_path: str,
                  yace_path: str,
                  prerun_steps: int,
                  max_steps: int,
                  n_cpu: int):
    """
    Run the LAMMPS benchmark.

    Args:
    - out_path: The path to the output directory.
    - yace_path: The path to the YACE executable.
    - prerun_steps: The number of pre-run steps.
    - max_steps: The maximum number of steps.
    - n_cpu: The number of CPUs to use.
    """
    lammps_in_values: dict = {
        'yace_path': yace_path
    }
    with open(LAMMPS_IN_TEMPLATE_PATH, 'r', encoding='utf-8') as file_template:
        lammps_in_template: Template = Template(file_template.read())
        lammps_in_content: str = lammps_in_template.substitute(lammps_in_values)
        lammps_in_out_path: str = os.path.join(out_path, 'lammps.in')
        with open(lammps_in_out_path, 'w', encoding='utf-8') as file_out:
            file_out.write(lammps_in_content)

    bench_script_values: dict = {
        'prerun_steps': prerun_steps,
        'max_steps': max_steps,
        'n_cpu': n_cpu,
        'lammps_bin_path': LAMMPS_BIN_PATH,
        'bench_potential_in_path': lammps_in_out_path
    }
    with open(BENCH_SCRIPT_TEMPLATE_PATH, 'r', encoding='utf-8') as file_template:
        bench_script_template: Template = Template(file_template.read())
        bench_script_content: str = bench_script_template.substitute(bench_script_values)
        bench_script_out_path: str = os.path.join(out_path, 'bench.sh')
        with open(bench_script_out_path, 'w', encoding='utf-8') as file_out:
            file_out.write(bench_script_content)

    result = subprocess.run(['bash', bench_script_out_path], check=True, capture_output=True, text=True)
    output_lines = result.stdout.splitlines()
    runtime3 = output_lines[-2]  # Assuming runtime3 is the second last line

    print(f"runtime3: {runtime3}")
