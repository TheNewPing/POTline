"""
Data analysis for LAMMPS simulations.
"""

import subprocess
from pathlib import Path

from ..config_reader import unpatify, gen_from_template

def run_properties_simulation(out_path: Path,
                              yace_path:Path,
                              lammps_bin_path: Path,
                              submit_template_path: Path,
                              lammps_inps_path: Path,
                              pps_python_path: Path,
                              ref_data_path: Path):
    """
    Run the properties simulation using LAMMPS.

    Args:
        out_path (Path): The path to the output directory.
        yace_path (Path): The path to the YACE executable.
        submit_template_path (Path): The path to the submit script template.
        lammps_bin_path (Path): The path to the LAMMPS binary.
        lammps_inps_path (Path): The path to the LAMMPS input files.
        pps_python_path (Path): The path to the Python script for post-processing.
        ref_data_path (Path): The path to the reference data.
    """
    simulation_values: dict = unpatify({
        'lammps_bin_path': lammps_bin_path,
        'lammps_inps_path': lammps_inps_path,
        'pps_python_path': pps_python_path,
        'ref_data_path': ref_data_path,
        'yace_path': yace_path,
        'out_path': yace_path.parent,
    })
    simulation_script_out_path: Path = out_path / 'submit.sh'
    gen_from_template(submit_template_path, simulation_values, simulation_script_out_path)

    subprocess.run(['bash', simulation_script_out_path], check=True)
