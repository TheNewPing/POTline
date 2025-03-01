"""
Properties simulation.
"""

from pathlib import Path
import shutil

from ..config_reader import ConfigReader
from ..loss_logger import ModelTracker
from ..dispatcher import DispatcherManager, JobType
from ..model import get_lammps_params

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

    def __init__(self, config_path: Path, tracker_list: list[ModelTracker]):
        self._config = ConfigReader(config_path).get_prop_config()
        self._tracker_list = tracker_list
        self._out_path = self._config.sweep_path / PROPERTIES_BENCH_DIR_NAME

    def prep_sim(self) -> None:
        """
        Prepare the simulation directories.
        """
        self._out_path.mkdir(exist_ok=True)

        for i, tracker in enumerate(self._tracker_list):
            iter_path = self._out_path / str(i+1)
            iter_path.mkdir(exist_ok=True)
            shutil.copy(SUBMIT_TEMPLATE_PATH, iter_path)
            shutil.copy(tracker.model.get_pot_path(), iter_path)
            tracker.save_info(iter_path)

    @staticmethod
    def run_sim(config_path: Path, dependency: int | None = None) -> int:
        """
        Run properties simulation.

        Args:
            - config_path: the path to the configuration file.
            - dependency: the job dependency.

        Returns:
            int: The id of the last watcher job.
        """
        sim_config = ConfigReader(config_path).get_prop_config()
        gen_config = ConfigReader(config_path).get_general_config()
        cli_path: Path = gen_config.repo_path / 'src' / 'run_sim.py'
        out_path: Path = sim_config.sweep_path / PROPERTIES_BENCH_DIR_NAME
        out_path.mkdir(exist_ok=True)
        watch_manager = DispatcherManager(
            JobType.WATCH_SIM.value, sim_config.model_name, sim_config.job_config.cluster)
        sim_manager = DispatcherManager(
            JobType.SIM.value, sim_config.model_name, sim_config.job_config.cluster)

        # init job
        init_cmd: str = f'{gen_config.python_bin} {cli_path} --config {config_path}'
        watch_manager.set_job([init_cmd], out_path, sim_config.job_config, dependency=dependency)
        init_id = watch_manager.dispatch_job()

        # run jobs
        cpus_per_task = int(sim_config.job_config.slurm_opts['cpus_per_task'])
        ntasks = int(sim_config.job_config.slurm_opts['ntasks'])
        sim_cmd: str = ' '.join([str(cmd) for cmd in [
            'bash', SUBMIT_SCRIPT_NAME,
            f'"{sim_config.lammps_bin_path} {get_lammps_params(sim_config.model_name)}"',
            PropertiesSimulator.LAMMPS_INPS_PATH,
            PropertiesSimulator.PPS_PYTHON_PATH,
            PropertiesSimulator.REF_DATA_PATH,
            cpus_per_task, ntasks
        ]])
        sim_manager.set_job([sim_cmd], out_path, sim_config.job_config, dependency=init_id,
                            array_ids=list(range(1, sim_config.best_n_models+1)))
        return sim_manager.dispatch_job()
