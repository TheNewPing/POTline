"""
Data analysis for LAMMPS simulations.
"""

import subprocess
from pathlib import Path

from ..utils import unpatify, gen_from_template

PROPERTIES_BENCH_DIR_NAME: str = 'properties_bench'
SUBMIT_TEMPLATE_NAME: str = 'submit_local.sh'
SUBMIT_TEMPLATE_PATH: Path = Path(__file__).parent / 'template' / SUBMIT_TEMPLATE_NAME

def run_properties_simulation(out_path: Path,
                              lammps_bin_path: Path,
                              lammps_inps_path: Path,
                              pps_python_path: Path,
                              ref_data_path: Path):
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
    simulation_script_out_path: Path = prop_bench_dir / SUBMIT_TEMPLATE_NAME
    gen_from_template(SUBMIT_TEMPLATE_PATH, simulation_values, simulation_script_out_path)

    subprocess.run(['bash', str(simulation_script_out_path)], check=True)
