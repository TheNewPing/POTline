"""
Data analysis for LAMMPS simulations.
"""

import os
import subprocess
from string import Template

SUBMIT_SCRIPT_TEMPLATE_PATH = ''

def run_properties_simulation(out_path: str,
                              yace_path:str,
                              lammps_bin_path: str,
                              lammps_inps_path: str,
                              pps_python_path: str):
    """
    Run the properties simulation using LAMMPS.

    Args:
        out_path (str): The path to the output directory.
        yace_path (str): The path to the YACE executable.
        lammps_bin_path (str): The path to the LAMMPS binary.
        lammps_inps_path (str): The path to the LAMMPS input files.
        pps_python_path (str): The path to the Python script for post-processing.
    """
    simulation_values: dict = {
        'lammps_bin_path': lammps_bin_path,
        'lammps_inps_path': lammps_inps_path,
        'pps_python_path': pps_python_path,
        'yace_path': yace_path
    }
    with open(SUBMIT_SCRIPT_TEMPLATE_PATH, 'r', encoding='utf-8') as file_template:
        simulation_script_template: Template = Template(file_template.read())
        simulation_script_content: str = simulation_script_template.substitute(simulation_values)
        simulation_script_out_path: str = os.path.join(out_path, 'submit.sh')
        with open(simulation_script_out_path, 'w', encoding='utf-8') as file_out:
            file_out.write(simulation_script_content)

    subprocess.run(['bash', simulation_script_out_path], check=True)
