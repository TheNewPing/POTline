"""
Properties simulation.
"""

from pathlib import Path

from ..experiment import Experiment

from ...config_reader import ConfigReader, MainSectionKW
from ...model import get_lammps_params

DISLOCATIONS_DIR_NAME: str = 'dislocations'
SUBMIT_SCRIPT_NAME: str = 'submit.sh'
DISLOCATIONS_TEMPLATE_PATH: Path = Path(__file__).parent / 'template'
SUBMIT_TEMPLATE_PATH: Path = DISLOCATIONS_TEMPLATE_PATH / SUBMIT_SCRIPT_NAME

class Dislocater():
    """
    Class for running the LAMMPS properties simulations.

    Args:
        - config_path: the path to the configuration file.
    """

    EXP_LIST: list[str] = ['edge_011_100', 'edge_011_111', 'edge_100_010', 'M111', 'screw']

    def __init__(self, config_path: Path):
        self._config_path = config_path
        self._config = ConfigReader(config_path).get_experiment_config(MainSectionKW.DISCLOCATIONS.value)
        self._out_path = self._config.sweep_path / DISLOCATIONS_DIR_NAME

    def run_sim(self, dependency: int | None = None) -> list[int]:
        """
        Run properties simulation.

        Args:
            - config_path: the path to the configuration file.
            - dependency: the job dependency.

        Returns:
            int: The id of the last watcher job.
        """
        dsl_cmd: str = ' '.join([str(cmd) for cmd in [
            'bash', SUBMIT_SCRIPT_NAME,
            f'"{self._config.lammps_bin_path} {get_lammps_params(self._config.model_name)}"',
        ]])

        self._out_path.mkdir(exist_ok=True)
        out_ids: list[int] = []
        for exp in self.EXP_LIST:
            out_ids.append(Experiment.run_exp(self._config_path, self._out_path / exp,
                                              DISLOCATIONS_TEMPLATE_PATH / exp,
                                              dsl_cmd, self._config.best_n_models,
                                              self._config.job_config, self._config.model_name, dependency))

        return out_ids
