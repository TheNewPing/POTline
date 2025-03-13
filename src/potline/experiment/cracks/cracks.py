"""
Properties simulation.
"""

from pathlib import Path

from ..experiment import Experiment

from ...config_reader import ConfigReader, MainSectionKW
from ...model import get_lammps_params

CRACKS_DIR_NAME: str = 'cracks'
SUBMIT_SCRIPT_NAME: str = 'submit.sh'
CRACKS_TEMPLATE_PATH: Path = Path(__file__).parent / 'template'

class Cracker():
    """
    Class for running the LAMMPS properties simulations.

    Args:
        - config_path: the path to the configuration file.
    """

    EXP_LIST: list[str] = ['coeff', 'CrackSystem_1', 'CrackSystem_2', 'CrackSystem_3', 'CrackSystem_4']

    def __init__(self, config_path: Path):
        self._config_path = config_path
        self._config = ConfigReader(config_path).get_experiment_config(MainSectionKW.CRACKS.value)
        self._out_path = self._config.sweep_path / CRACKS_DIR_NAME

    def run_sim(self, dependency: int | None = None) -> list[int]:
        """
        Run properties simulation.

        Args:
            - config_path: the path to the configuration file.
            - dependency: the job dependency.

        Returns:
            int: The id of the last watcher job.
        """

        coeff_cmd: str = ' '.join([str(cmd) for cmd in ['bash', SUBMIT_SCRIPT_NAME]])

        self._out_path.mkdir(exist_ok=True)
        setup_id: int = Experiment.run_exp(self._config_path, self._out_path / self.EXP_LIST[0],
                                           CRACKS_TEMPLATE_PATH / self.EXP_LIST[0],
                                           coeff_cmd, self._config.best_n_models,
                                           self._config.job_config, self._config.model_name, dependency)

        cracks_cmd: str = ' '.join([str(cmd) for cmd in [
            'bash', SUBMIT_SCRIPT_NAME,
            f'"{self._config.lammps_bin_path} {get_lammps_params(self._config.model_name)}"',
        ]])

        out_ids: list[int] = []
        for exp in self.EXP_LIST[1:]:
            out_ids.append(Experiment.run_exp(self._config_path, self._out_path / exp,
                                              CRACKS_TEMPLATE_PATH / exp,
                                              cracks_cmd, self._config.best_n_models,
                                              self._config.job_config, self._config.model_name, setup_id))

        return out_ids
