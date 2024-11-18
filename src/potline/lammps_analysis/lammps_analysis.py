"""
Data analysis for LAMMPS simulations.
"""

import subprocess
from string import Template
from pathlib import Path

from ..config_reader import unpatify

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
    """
    simulation_values: dict = unpatify({
        'lammps_bin_path': lammps_bin_path,
        'lammps_inps_path': lammps_inps_path,
        'pps_python_path': pps_python_path,
        'ref_data_path': ref_data_path,
        'yace_path': yace_path
    })
    with submit_template_path.open('r', encoding='utf-8') as file_template:
        simulation_script_template: Template = Template(file_template.read())
        simulation_script_content: str = simulation_script_template.safe_substitute(simulation_values)
        simulation_script_out_path: Path = out_path / 'submit.sh'
        with simulation_script_out_path.open('w', encoding='utf-8') as file_out:
            file_out.write(simulation_script_content)

    subprocess.run(['bash', simulation_script_out_path], check=True)
