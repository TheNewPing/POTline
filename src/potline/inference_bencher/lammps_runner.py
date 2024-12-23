"""
This module is responsible for running LAMMPS benchmarks.
"""

from pathlib import Path
import shutil

from ..config_reader import ConfigReader
from ..loss_logger import ModelTracker

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
        - config: configuration for the inference benchmark
        - tracker_list: models to benchmark
    """
    def __init__(self, config_path: Path, tracker_list: list[ModelTracker]):
        self._config = ConfigReader(config_path).get_bench_config()
        self._tracker_list = tracker_list
        self._out_path = self._config.sweep_path / INFERENCE_BENCH_DIR_NAME

    def prep_inf(self) -> None:
        """
        Prepare the inference benchmark.
        """
        self._out_path.mkdir(exist_ok=True)

        for i, tracker in enumerate(self._tracker_list):
            iter_path = self._out_path / str(i+1)
            iter_path.mkdir(exist_ok=True)
            shutil.copy(LAMMPS_IN_PATH, iter_path)
            shutil.copy(BENCH_SCRIPT_TEMPLATE_PATH, iter_path)
            shutil.copy(tracker.model.get_pot_path(), iter_path)
            tracker.save_info(iter_path)
