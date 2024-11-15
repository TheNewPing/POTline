import os
import subprocess
from string import Template

LAMMPS_BIN_PATH = ''
LAMMPS_IN_TEMPLATE_PATH = './template/bench.in'
BENCH_SCRIPT_TEMPLATE_PATH = './template/run.sh'

def run_benchmark(out_path: str, yace_path: str, prerun_steps: int, max_steps: int, n_cpu: int):
    lammps_in_values: dict = {
        'yace_path': yace_path
    }
    with open(LAMMPS_IN_TEMPLATE_PATH, 'r') as file_template:
        lammps_in_template: Template = Template(file_template.read())
        lammps_in_content: str = lammps_in_template.substitute(lammps_in_values)
        lammps_in_out_path: str = os.path.join(out_path, 'lammps.in')
        with open(lammps_in_out_path, 'w') as file_out:
            file_out.write(lammps_in_content)

    bench_script_values: dict = {
        'prerun_steps': prerun_steps,
        'max_steps': max_steps,
        'n_cpu': n_cpu,
        'lammps_bin_path': LAMMPS_BIN_PATH,
        'bench_potential_in_path': lammps_in_out_path
    }
    with open(BENCH_SCRIPT_TEMPLATE_PATH, 'r') as file_template:
        bench_script_template: Template = Template(file_template.read())
        bench_script_content: str = bench_script_template.substitute(bench_script_values)
        bench_script_out_path: str = os.path.join(out_path, 'bench.sh')
        with open(bench_script_out_path, 'w') as file_out:
            file_out.write(bench_script_content)

    result = subprocess.run(['bash', bench_script_out_path], capture_output=True, text=True)
    output_lines = result.stdout.splitlines()
    runtime3 = output_lines[-2]  # Assuming runtime3 is the second last line

    print(f"runtime3: {runtime3}")