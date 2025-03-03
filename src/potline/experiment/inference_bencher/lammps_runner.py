"""
This module is responsible for running LAMMPS benchmarks.
"""

from pathlib import Path

from ..experiment import Experiment
from ...config_reader import ConfigReader
from ...model import get_lammps_params

INFERENCE_BENCH_DIR_NAME: str = 'inference_bench'
LAMMPS_IN_NAME: str = 'bench.in'
BENCH_SCRIPT_NAME: str = 'run.sh'
INF_BENCH_TEMPLATE_PATH: Path = Path(__file__).parent / 'template'
LAMMPS_IN_PATH: Path =  INF_BENCH_TEMPLATE_PATH / LAMMPS_IN_NAME
BENCH_SCRIPT_TEMPLATE_PATH: Path = INF_BENCH_TEMPLATE_PATH / BENCH_SCRIPT_NAME

class InferenceBencher():
    """
    Class for running the LAMMPS inference benchmark.

    Args:
        - config_path: the path to the configuration file.
    """
    def __init__(self, config_path: Path):
        self._config_path = config_path
        self._config = ConfigReader(config_path).get_bench_config()
        self._out_path = self._config.experiment_config.sweep_path / INFERENCE_BENCH_DIR_NAME

    def run_inf(self, dependency: int | None = None) -> int:
        """
        Run inference benchmark.

        Args:
            - config_path: the path to the configuration file.
            - dependency: the job dependency.

        Returns:
            int: The id of the last watcher job.
        """
        bench_cmd: str = ' '.join([str(cmd) for cmd in [
            'bash', BENCH_SCRIPT_NAME,
            self._config.experiment_config.lammps_bin_path,
            get_lammps_params(self._config.experiment_config.model_name),
            self._config.prerun_steps, self._config.max_steps,
        ]])

        return Experiment.run_exp(self._config_path, self._out_path, INF_BENCH_TEMPLATE_PATH,
                                  INFERENCE_BENCH_DIR_NAME, bench_cmd,
                                  self._config.experiment_config.best_n_models,
                                  self._config.experiment_config.job_config,
                                  self._config.experiment_config.model_name, dependency)
