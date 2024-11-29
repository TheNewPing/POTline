"""
This module is responsible for running LAMMPS benchmarks.
"""

import subprocess
from pathlib import Path

from simple_slurm import Slurm

from ..utils import unpatify, gen_from_template, POTENTIAL_NAME

INFERENCE_BENCH_DIR_NAME: str = 'inference_bench'
LAMMPS_IN_NAME: str = 'bench.in'
BENCH_SCRIPT_NAME: str = 'run.sh'
BENCH_HPC_SCRIPT_NAME: str = 'run_hpc.txt'
INF_BENCH_TEMPLATE_PATH: Path = Path(__file__).parent / 'template'
LAMMPS_IN_TEMPLATE_PATH: Path =  INF_BENCH_TEMPLATE_PATH / LAMMPS_IN_NAME
BENCH_SCRIPT_TEMPLATE_PATH: Path = INF_BENCH_TEMPLATE_PATH / BENCH_SCRIPT_NAME
BENCH_HPC_SCRIPT_TEMPLATE_PATH: Path = INF_BENCH_TEMPLATE_PATH / BENCH_HPC_SCRIPT_NAME

def run_benchmark(out_path: Path,
                  lammps_bin_path: Path,
                  prerun_steps: int,
                  max_steps: int,
                  n_cpu: int,
                  hpc: bool = False):
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

    lammps_in_values: dict = unpatify({
        'pot_path': out_path / POTENTIAL_NAME,
    })
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

    if not hpc:
        subprocess.run(['bash', str(bench_script_out_path)], check=True)
    else:
        bench_job: Slurm = Slurm(
            job_name='lammps_inf_bench',
            n_tasks=1,
            n_cpu=n_cpu,
            time_limit='03:00:00',
            stderr_path=inf_bench_dir / 'slurm_inf.stderr',
            stdout_path=inf_bench_dir / 'slurm_inf.stdout',
        )
        bench_job.add_cmd('module load 2022')
        bench_job.add_cmd('module load OpenMPI/4.1.4-NVHPC-22.7-CUDA-11.7.0')
        bench_job.sbatch(f'bash {bench_script_out_path}')
