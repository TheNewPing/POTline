"""
This module is responsible for running LAMMPS benchmarks.
"""

import subprocess
from pathlib import Path

from ..utils import unpatify, gen_from_template

INFERENCE_BENCH_DIR_NAME: str = 'inference_bench'
LAMMPS_IN_NAME: str = 'bench.in'
BENCH_SCRIPT_NAME: str = 'run.sh'
LAMMPS_IN_TEMPLATE_PATH: Path = Path(__file__).parent / 'template' / LAMMPS_IN_NAME
BENCH_SCRIPT_TEMPLATE_PATH: Path = Path(__file__).parent / 'template' / BENCH_SCRIPT_NAME

def run_benchmark(out_path: Path,
                  lammps_bin_path: Path,
                  prerun_steps: int,
                  max_steps: int,
                  n_cpu: int):
    """
    Run the LAMMPS benchmark.

    Args:
    - out_path: The path to the output directory.
    - yace_path: The path to the YACE executable.
    - lammps_bin_path: The path to the LAMMPS binary.
    - prerun_steps: The number of pre-run steps.
    - max_steps: The maximum number of steps.
    - n_cpu: The number of CPUs to use.
    """
    inf_bench_dir: Path = out_path / INFERENCE_BENCH_DIR_NAME
    inf_bench_dir.mkdir(exist_ok=True)

    lammps_in_values: dict = unpatify({})
    lammps_in_out_path: Path = inf_bench_dir / LAMMPS_IN_NAME
    gen_from_template(LAMMPS_IN_TEMPLATE_PATH, lammps_in_values, lammps_in_out_path)

    bench_script_values: dict = unpatify({
        'prerun_steps': prerun_steps,
        'max_steps': max_steps,
        'n_cpu': n_cpu,
        'lammps_bin_path': lammps_bin_path,
        'bench_potential_in_path': lammps_in_out_path,
        'out_path': inf_bench_dir
    })
    bench_script_out_path: Path = inf_bench_dir / BENCH_SCRIPT_NAME
    gen_from_template(BENCH_SCRIPT_TEMPLATE_PATH, bench_script_values, bench_script_out_path)

    subprocess.run(['bash', bench_script_out_path], check=True)
