"""
This module is responsible for running LAMMPS benchmarks.
"""

from pathlib import Path
import shutil

from ..config_reader import BenchConfig
from ..dispatcher import DispatcherManager
from ..loss_logger import ModelTracker

INFERENCE_BENCH_DIR_NAME: str = 'inference_bench'
LAMMPS_IN_NAME: str = 'bench.in'
BENCH_SCRIPT_NAME: str = 'run.sh'
INF_BENCH_TEMPLATE_PATH: Path = Path(__file__).parent / 'template'
LAMMPS_IN_PATH: Path =  INF_BENCH_TEMPLATE_PATH / LAMMPS_IN_NAME
BENCH_SCRIPT_TEMPLATE_PATH: Path = INF_BENCH_TEMPLATE_PATH / BENCH_SCRIPT_NAME

_N_CPU: int = 1

class InferenceBencher():
    """
    Class for running the LAMMPS inference benchmark.

    Args:
        - config: configuration for the benchmark
        - tracker_list: trackers to benchmark
        - dispatcher_manager: manager for dispatching benchmark jobs
    """
    def __init__(self, config: BenchConfig, tracker_list: list[ModelTracker],
                 dispatcher_manager: DispatcherManager):
        self._config = config
        self._tracker_list = tracker_list
        self._dispatcher_manager = dispatcher_manager
        self._lammps_params = tracker_list[0].model.get_lammps_params()
        self._out_path = self._config.sweep_path / INFERENCE_BENCH_DIR_NAME
        self._out_path.mkdir(exist_ok=True)

        self._bench_cmd: str = ' '.join(
            [str(cmd) for cmd in [
                'srun', BENCH_SCRIPT_NAME, _N_CPU,
                f'"{config.lammps_bin_path} {self._lammps_params}"',
                config.prerun_steps, config.max_steps]])

        for i, tracker in enumerate(self._tracker_list):
            iter_path = self._out_path / str(i)
            iter_path.mkdir(exist_ok=True)
            shutil.copy(LAMMPS_IN_PATH, iter_path)
            shutil.copy(BENCH_SCRIPT_TEMPLATE_PATH, iter_path)
            shutil.copy(tracker.model.get_pot_path(), iter_path)
            tracker.save_info(iter_path)

    def run(self):
        self._dispatcher_manager.set_job([self._bench_cmd],
                                         self._out_path,
                                         self._config.job_config,
                                         list(range(1,len(self._tracker_list)+1)))
        self._dispatcher_manager.dispatch_job()
