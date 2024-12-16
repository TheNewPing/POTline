"""
Properties simulation.
"""

from pathlib import Path

from ..config_reader import PropConfig
from ..dispatcher import DispatcherFactory, Dispatcher
from ..model import PotModel

PROPERTIES_BENCH_DIR_NAME: str = 'properties_bench'
SUBMIT_SCRIPT_NAME: str = 'submit.sh'
PROP_BENCH_TEMPLATE_PATH: Path = Path(__file__).parent / 'template'
SUBMIT_TEMPLATE_PATH: Path = PROP_BENCH_TEMPLATE_PATH / SUBMIT_SCRIPT_NAME

_N_CPU: int = 1

def run_properties_simulation(model: PotModel, config: PropConfig,
                              dispatcher_factory: DispatcherFactory):
    """
    Run the properties simulation using LAMMPS.

    Args:
        - model: model to use for the simulation
        - config: properties simulation configuration
        - dispatcher_factory: factory for dispatching the simulation
    """
    prop_bench_dir: Path = model.get_out_path().parent / PROPERTIES_BENCH_DIR_NAME
    prop_bench_dir.mkdir(exist_ok=True)

    command: list[str] = [str(cmd) for cmd in
                          ['bash', SUBMIT_TEMPLATE_PATH, prop_bench_dir,
                           f'"{config.lammps_bin_path} {model.get_lammps_params()}"',
                           config.lammps_inps_path, config.pps_python_path, config.ref_data_path,
                           config.email, _N_CPU]]

    dispatcher: Dispatcher = dispatcher_factory.create_dispatcher(
        [' '.join(command)], prop_bench_dir, model=model.get_name().value, n_cpu=_N_CPU, email=config.email)
    dispatcher.dispatch()
