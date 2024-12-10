"""
Data analysis for LAMMPS simulations.
"""

from pathlib import Path

from ..config_reader import PropConfig
from ..dispatcher import DispatcherFactory, Dispatcher

PROPERTIES_BENCH_DIR_NAME: str = 'properties_bench'
SUBMIT_SCRIPT_NAME: str = 'submit.sh'
PROP_BENCH_TEMPLATE_PATH: Path = Path(__file__).parent / 'template'
SUBMIT_TEMPLATE_PATH: Path = PROP_BENCH_TEMPLATE_PATH / SUBMIT_SCRIPT_NAME

def run_properties_simulation(fitted_path: Path, config: PropConfig,
                              dispatcher_factory: DispatcherFactory):
    """
    Run the properties simulation using LAMMPS.

    Args:
        - fitted_path: path to the directory with the fitted models.
        - config: configuration for the data analysis.
    """
    prop_bench_dir: Path = fitted_path / PROPERTIES_BENCH_DIR_NAME
    prop_bench_dir.mkdir(exist_ok=True)

    command: list[str] = [str(cmd) for cmd in
                          ['bash', SUBMIT_TEMPLATE_PATH, prop_bench_dir, config.lammps_bin_path,
                           config.lammps_inps_path, config.pps_python_path, config.ref_data_path,
                           config.email]]

    dispatcher: Dispatcher = dispatcher_factory.create_dispatcher(command, prop_bench_dir, email=config.email)
    dispatcher.dispatch()
