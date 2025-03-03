"""
Properties simulation.
"""

from pathlib import Path

from ..experiment import Experiment

from ...config_reader import ConfigReader, MainSectionKW
from ...model import get_lammps_params

HSS_DIR_NAME: str = 'hard_split_screw'
SUBMIT_SCRIPT_NAME: str = 'submit.sh'
HSS_TEMPLATE_PATH: Path = Path(__file__).parent / 'template'
SUBMIT_TEMPLATE_PATH: Path = HSS_TEMPLATE_PATH / SUBMIT_SCRIPT_NAME

class HardSplitter():
    """
    Class for running the LAMMPS properties simulations.

    Args:
        - config_path: the path to the configuration file.
    """

    DB_PATH: Path = Path(__file__).parent / 'screw_db'

    def __init__(self, config_path: Path):
        self._config_path = config_path
        self._config = ConfigReader(config_path).get_experiment_config(MainSectionKW.HARD_SPLIT_SCREW.value)
        self._out_path = self._config.sweep_path / HSS_DIR_NAME

    def run_sim(self, dependency: int | None = None) -> int:
        """
        Run properties simulation.

        Args:
            - config_path: the path to the configuration file.
            - dependency: the job dependency.

        Returns:
            int: The id of the last watcher job.
        """
        hss_cmd: str = ' '.join([str(cmd) for cmd in [
            'bash', SUBMIT_SCRIPT_NAME,
            f'"{self._config.lammps_bin_path} {get_lammps_params(self._config.model_name)}"',
            HardSplitter.DB_PATH,
        ]])

        return Experiment.run_exp(self._config_path, self._out_path, HSS_TEMPLATE_PATH,
                                  HSS_DIR_NAME, hss_cmd, self._config.best_n_models,
                                  self._config.job_config, self._config.model_name, dependency)
