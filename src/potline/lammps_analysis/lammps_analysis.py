"""
Data analysis for LAMMPS simulations.
"""

import subprocess
from pathlib import Path

from simple_slurm import Slurm # type: ignore

from ..config_reader import PropConfig

PROPERTIES_BENCH_DIR_NAME: str = 'properties_bench'
SUBMIT_SCRIPT_NAME: str = 'submit.sh'
PROP_BENCH_TEMPLATE_PATH: Path = Path(__file__).parent / 'template'
SUBMIT_TEMPLATE_PATH: Path = PROP_BENCH_TEMPLATE_PATH / SUBMIT_SCRIPT_NAME
EMAIL: str = 'e.rodaro@rug.nl'

def run_properties_simulation(fitted_path: Path, config: PropConfig, hpc: bool = False) -> int:
    """
    Run the properties simulation using LAMMPS.

    Args:
        - fitted_path: path to the directory with the fitted models.
        - config: configuration for the data analysis.
        - hpc: flag to run the simulations on HPC.

    Returns:
        int: 0 if the simulation is run locally, the job id if it is run on the HPC.
    """
    prop_bench_dir: Path = fitted_path / PROPERTIES_BENCH_DIR_NAME
    prop_bench_dir.mkdir(exist_ok=True)

    command: list[str] = [str(cmd) for cmd in
                          ['bash', SUBMIT_TEMPLATE_PATH, prop_bench_dir, hpc, config.lammps_bin_path,
                           config.lammps_inps_path, config.pps_python_path, config.ref_data_path, EMAIL]]

    if not hpc:
        subprocess.run(command, check=True)
        return 0

    prop_job: Slurm = Slurm(
        job_name='lammps_prop_bench',
        ntasks=32,
        cpus_per_task=1,
        mem='10G',
        time='3:00:00',
        error=prop_bench_dir / 'prop_%j.stderr',
        output=prop_bench_dir / 'prop_%j.stdout',
        mail_type='ALL',
        mail_user=EMAIL
    )
    return prop_job.sbatch(' '.join(command))
