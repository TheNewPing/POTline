"""
This module is responsible for running LAMMPS benchmarks.
"""

import subprocess
from string import Template
from pathlib import Path

from ..config_reader import unpatify

def run_benchmark(out_path: Path,
                  yace_path: Path,
                  lammps_bin_path: Path,
                  lammps_inps_template_path: Path,
                  bench_script_template_path: Path,
                  prerun_steps: int,
                  max_steps: int,
                  n_cpu: int):
    """
    Run the LAMMPS benchmark.

    Args:
    - out_path: The path to the output directory.
    - yace_path: The path to the YACE executable.
    - lammps_bin_path: The path to the LAMMPS binary.
    - lammps_inps_template_path: The path to the LAMMPS input template.
    - bench_script_template_path: The path to the benchmark script template.
    - prerun_steps: The number of pre-run steps.
    - max_steps: The maximum number of steps.
    - n_cpu: The number of CPUs to use.
    """
    lammps_in_values: dict = unpatify({
        'yace_path': yace_path
    })
    with lammps_inps_template_path.open('r', encoding='utf-8') as file_template:
        lammps_in_template: Template = Template(file_template.read())
        lammps_in_content: str = lammps_in_template.safe_substitute(lammps_in_values)
        lammps_in_out_path: Path = out_path / 'lammps.in'
        with open(lammps_in_out_path, 'w', encoding='utf-8') as file_out:
            file_out.write(lammps_in_content)

    bench_script_values: dict = unpatify({
        'prerun_steps': prerun_steps,
        'max_steps': max_steps,
        'n_cpu': n_cpu,
        'lammps_bin_path': lammps_bin_path,
        'bench_potential_in_path': lammps_in_out_path
    })
    with bench_script_template_path.open('r', encoding='utf-8') as file_template:
        bench_script_template: Template = Template(file_template.read())
        bench_script_content: str = bench_script_template.safe_substitute(bench_script_values)
        bench_script_out_path: Path = out_path / 'bench.sh'
        with open(bench_script_out_path, 'w', encoding='utf-8') as file_out:
            file_out.write(bench_script_content)

    result = subprocess.run(['bash', bench_script_out_path], check=True, capture_output=True, text=True)
    output_lines = result.stdout.splitlines()
    runtime3 = output_lines[-2]  # Assuming runtime3 is the second last line

    print(f"runtime3: {runtime3}")
