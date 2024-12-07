"""
This module is responsible for running LAMMPS benchmarks.
"""

import subprocess
from pathlib import Path

from simple_slurm import Slurm # type: ignore

from ..config_reader import BenchConfig

INFERENCE_BENCH_DIR_NAME: str = 'inference_bench'
LAMMPS_IN_NAME: str = 'bench.in'
BENCH_SCRIPT_NAME: str = 'run.sh'
INF_BENCH_TEMPLATE_PATH: Path = Path(__file__).parent / 'template'
LAMMPS_IN_PATH: Path =  INF_BENCH_TEMPLATE_PATH / LAMMPS_IN_NAME
BENCH_SCRIPT_TEMPLATE_PATH: Path = INF_BENCH_TEMPLATE_PATH / BENCH_SCRIPT_NAME

def run_benchmark(fitted_path: Path, config: BenchConfig, hpc: bool = False) -> int:
    """
    Run the LAMMPS benchmark.

    Args:
        - fitted_path: path to the directory with the fitted models.
        - config: configuration for the inference benchmark.
        - hpc: flag to run the simulations on HPC.

    Returns:
        int: 0 if the benchmark is run locally, the job id if it is run on the HPC.
    """
    inf_bench_dir: Path = fitted_path / INFERENCE_BENCH_DIR_NAME
    inf_bench_dir.mkdir(exist_ok=True)

    command: list[str] = [str(cmd) for cmd in ['bash', BENCH_SCRIPT_TEMPLATE_PATH, config.n_cpu, hpc,
                                               config.lammps_bin_path, LAMMPS_IN_PATH, config.prerun_steps,
                                               config.max_steps, inf_bench_dir]]

    if not hpc:
        subprocess.run(command, check=True)
        return 0

    bench_job: Slurm = Slurm(
        job_name='lammps_inf_bench',
        ntasks=1,
        cpus_per_task=config.n_cpu,
        time='03:00:00',
        error=inf_bench_dir / 'inf_%j.err',
        output=inf_bench_dir / 'inf_%j.out'
    )
    return bench_job.sbatch(' '.join(command))
