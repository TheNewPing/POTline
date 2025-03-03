"""
Properties simulation.
"""

from pathlib import Path

from ..experiment import Experiment

from ...config_reader import ConfigReader
from ...dispatcher import JobType
from ...model import get_lammps_params

PROPERTIES_BENCH_DIR_NAME: str = 'properties_bench'
SUBMIT_SCRIPT_NAME: str = 'submit.sh'
PROP_BENCH_TEMPLATE_PATH: Path = Path(__file__).parent / 'template'
SUBMIT_TEMPLATE_PATH: Path = PROP_BENCH_TEMPLATE_PATH / SUBMIT_SCRIPT_NAME

class PropertiesSimulator():
    """
    Class for running the LAMMPS properties simulations.

    Args:
        - config : configuration for the simulations
        - tracker_list : trackers to simulate
    """

    LAMMPS_INPS_PATH: Path = Path(__file__).parent / 'pot_testing' / 'lmps_inputs'
    PPS_PYTHON_PATH: Path = Path(__file__).parent / 'pot_testing' / 'py_pps'
    REF_DATA_PATH: Path = Path(__file__).parent / 'pot_testing' / 'REF_DATA'

    def __init__(self, config_path: Path):
        self._config_path = config_path
        self._config = ConfigReader(config_path).get_prop_config()
        self._out_path = self._config.sweep_path / PROPERTIES_BENCH_DIR_NAME

    def run_sim(self, dependency: int | None = None) -> int:
        """
        Run properties simulation.

        Args:
            - config_path: the path to the configuration file.
            - dependency: the job dependency.

        Returns:
            int: The id of the last watcher job.
        """
        sim_cmd: str = ' '.join([str(cmd) for cmd in [
            'bash', SUBMIT_SCRIPT_NAME,
            f'"{self._config.lammps_bin_path} {get_lammps_params(self._config.model_name)}"',
            PropertiesSimulator.LAMMPS_INPS_PATH,
            PropertiesSimulator.PPS_PYTHON_PATH,
            PropertiesSimulator.REF_DATA_PATH,
        ]])

        return Experiment.run_exp(self._config_path, self._out_path, PROP_BENCH_TEMPLATE_PATH,
                                  PROPERTIES_BENCH_DIR_NAME, sim_cmd, self._config.best_n_models,
                                  self._config.job_config, JobType.WATCH_SIM.value, JobType.SIM.value,
                                  self._config.model_name, dependency)
