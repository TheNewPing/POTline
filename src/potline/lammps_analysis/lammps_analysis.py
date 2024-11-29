"""
Data analysis for LAMMPS simulations.
"""

import subprocess
from pathlib import Path

from simple_slurm import Slurm

from ..utils import unpatify, gen_from_template

PROPERTIES_BENCH_DIR_NAME: str = 'properties_bench'
SUBMIT_TEMPLATE_NAME: str = 'submit_local.sh'
SUBMIT_HPC_TEMPLATE_NAME: str = 'submit_hpc.sh'
PROP_BENCH_TEMPLATE_PATH: Path = Path(__file__).parent / 'template'
SUBMIT_TEMPLATE_PATH: Path = PROP_BENCH_TEMPLATE_PATH / SUBMIT_TEMPLATE_NAME
SUBMIT_HPC_TEMPLATE_PATH: Path = PROP_BENCH_TEMPLATE_PATH / SUBMIT_HPC_TEMPLATE_NAME

def run_properties_simulation(out_path: Path,
                              lammps_bin_path: Path,
                              lammps_inps_path: Path,
                              pps_python_path: Path,
                              ref_data_path: Path,
                              hpc: bool = False):
    """
    Run the properties simulation using LAMMPS.

    Args:
        out_path (Path): The path to the output directory.
        yace_path (Path): The path to the YACE executable.
        lammps_bin_path (Path): The path to the LAMMPS binary.
        lammps_inps_path (Path): The path to the LAMMPS input files.
        pps_python_path (Path): The path to the Python script for post-processing.
        ref_data_path (Path): The path to the reference data.
    """
    prop_bench_dir: Path = out_path / PROPERTIES_BENCH_DIR_NAME
    prop_bench_dir.mkdir(exist_ok=True)

    simulation_values: dict = unpatify({
        'lammps_bin_path': lammps_bin_path,
        'lammps_inps_path': lammps_inps_path,
        'pps_python_path': pps_python_path,
        'ref_data_path': ref_data_path,
        'out_path': prop_bench_dir,
    })

    command: list[str] = [str(cmd) for cmd in
                          ['bash', prop_bench_dir, hpc, lammps_bin_path, lammps_inps_path, pps_python_path, ref_data_path]]

    if not hpc:
        subprocess.run(['bash', command], check=True)
    else:

        simulation_values.update(unpatify({
            'job_name': 'lammps_prop_bench',
            'n_tasks': 32,
            'n_cpu': 1,
            'time_limit': '2:00:00',
            'stderr_path': prop_bench_dir / 'slurm_prop.stderr',
            'stdout_path': prop_bench_dir / 'slurm_prop.stdout',
            'email': 'e.rodaro@rug.nl'
        }))
        simulation_script_out_path = prop_bench_dir / SUBMIT_HPC_TEMPLATE_NAME
        gen_from_template(SUBMIT_HPC_TEMPLATE_PATH, simulation_values, simulation_script_out_path)
        subprocess.run(['sbatch', str(simulation_script_out_path)], check=True)
