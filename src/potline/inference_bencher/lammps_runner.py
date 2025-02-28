"""
This module is responsible for running LAMMPS benchmarks.
"""

from pathlib import Path
import shutil

from ..config_reader import ConfigReader
from ..loss_logger import ModelTracker
from ..dispatcher import DispatcherManager, JobType
from ..model import get_lammps_params

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

    @staticmethod
    def run_inf(config_path: Path, dependency: int | None = None) -> int:
        """
        Run inference benchmark.

        Args:
            - config_path: the path to the configuration file.
            - dependency: the job dependency.

        Returns:
            int: The id of the last watcher job.
        """
        inf_config = ConfigReader(config_path).get_bench_config()
        gen_config = ConfigReader(config_path).get_general_config()
        cli_path: Path = gen_config.repo_path/ 'src' / 'run_inf.py'
        out_path: Path = inf_config.sweep_path / INFERENCE_BENCH_DIR_NAME
        out_path.mkdir(exist_ok=True)
        watch_manager = DispatcherManager(
            JobType.WATCH_INF.value, inf_config.model_name, inf_config.job_config.cluster)
        inf_manager = DispatcherManager(
            JobType.INF.value, inf_config.model_name, inf_config.job_config.cluster)

        # init job
        init_cmd: str = f'{gen_config.python_bin} {cli_path} --config {config_path}'
        watch_manager.set_job([init_cmd], out_path, inf_config.job_config, dependency=dependency)
        init_id = watch_manager.dispatch_job()

        # run jobs
        cpus_per_task = int(inf_config.job_config.slurm_opts['cpus_per_task'])
        ntasks = int(inf_config.job_config.slurm_opts['ntasks'])
        bench_cmd: str = ' '.join([str(cmd) for cmd in [
            'bash', BENCH_SCRIPT_NAME,
            f'"{inf_config.lammps_bin_path} {get_lammps_params(inf_config.model_name)}"',
            inf_config.prerun_steps, inf_config.max_steps,
            cpus_per_task, ntasks
        ]])
        inf_manager.set_job([bench_cmd], out_path, inf_config.job_config, dependency=init_id,
                            array_ids=list(range(1,inf_config.best_n_models+1)))
        return inf_manager.dispatch_job()
