"""
CLI script for dispatching PotLine.
"""

from argparse import Namespace, ArgumentParser
from pathlib import Path

from potline.dispatcher import DispatcherManager, JobType
from potline.config_reader import ConfigReader
from potline.model import get_fit_cmd, get_lammps_params
from potline.hyper_searcher import OPTIM_DIR_NAME
from potline.deep_trainer import DEEP_TRAIN_DIR_NAME
from potline.inference_bencher import BENCH_SCRIPT_NAME, INFERENCE_BENCH_DIR_NAME
from potline.properties_simulator import PROPERTIES_BENCH_DIR_NAME, SUBMIT_SCRIPT_NAME, PropertiesSimulator

PYTHON_BIN: str = 'conda run python'

def parse_args() -> Namespace:
    """
    Parse the command line arguments.
    """
    parser: ArgumentParser = ArgumentParser(description='Process some parameters.')
    parser.add_argument('--config', type=str, help='Path to the config file')
    parser.add_argument('--nohyper', action='store_false', help='Disable hyperparameter search')
    parser.add_argument('--hypiter', type=int, default=1, help='Hyperparameter search startibg iteration')
    parser.add_argument('--nodeep', action='store_false', help='Disable deep training')
    parser.add_argument('--noconversion', action='store_false', help='Disable yace conversion')
    parser.add_argument('--noinference', action='store_false', help='Disable inference benchmark')
    parser.add_argument('--noproperties', action='store_false', help='Disable properties simulation')
    return parser.parse_args()

def run_hyp(config_path: Path, start_iter: int) -> int:
    """
    Run hyperparameter search.

    Args:
        - config_path: the path to the configuration file.
        - start_iter: the starting iteration.
            If > 1, assusmes that iteration i-1 has already been registered.

    Returns:
        int: The id of the last watcher job.
    """
    # Dispatch watchers
    hyp_config = ConfigReader(config_path).get_optimizer_config()
    cli_path: Path = Path(__file__).resolve().parent / 'run_hyp.py'
    out_path: Path = hyp_config.sweep_path / OPTIM_DIR_NAME
    fit_manager = DispatcherManager(
        JobType.FIT.value, hyp_config.model_name, hyp_config.job_config.cluster)
    watch_manager = DispatcherManager(
        JobType.WATCH_FIT.value, hyp_config.model_name, hyp_config.job_config.cluster)

    # init job
    init_cmd: str = f'{PYTHON_BIN} {cli_path} --config {config_path} --iteration {start_iter}'
    watch_manager.set_job([init_cmd], out_path / str(start_iter), hyp_config.job_config)
    watch_id = watch_manager.dispatch_job()

    # run jobs
    fit_cmd: str = get_fit_cmd(hyp_config.model_name, deep=False)
    for i in range(start_iter, hyp_config.max_iter+1):
        fit_manager.set_job([fit_cmd], out_path / str(i), hyp_config.job_config, dependency=watch_id,
                            array_ids=list(range(1,hyp_config.n_points+1)))
        fit_id = fit_manager.dispatch_job()
        cmd: str = f'{PYTHON_BIN} {cli_path} --config {config_path} --restart --iteration {i+1}'
        watch_manager.set_job([cmd], out_path / str(i+1),
                                hyp_config.job_config, dependency=fit_id)
        watch_id = watch_manager.dispatch_job()
    return watch_id

def run_deep(config_path: Path, dependency: int | None = None) -> int:
    """
    Run deep training.

    Args:
        - config_path: the path to the configuration file.
        - dependency: the job dependency.

    Returns:
        int: The id of the last watcher job.
    """
    deep_config = ConfigReader(config_path).get_deep_train_config()
    cli_path: Path = Path(__file__).resolve().parent / 'run_deep.py'
    out_path: Path = deep_config.sweep_path / DEEP_TRAIN_DIR_NAME
    deep_manager = DispatcherManager(
        JobType.DEEP.value, deep_config.model_name, deep_config.job_config.cluster)
    watch_manager = DispatcherManager(
        JobType.WATCH_DEEP.value, deep_config.model_name, deep_config.job_config.cluster)

    # init job
    init_cmd: str = f'{PYTHON_BIN} {cli_path} --config {config_path}'
    watch_manager.set_job([init_cmd], out_path, deep_config.job_config, dependency=dependency)
    init_id = watch_manager.dispatch_job()

    # fit jobs
    deep_cmd: str = get_fit_cmd(deep_config.model_name, deep=True)
    deep_manager.set_job([deep_cmd], out_path, deep_config.job_config, dependency=init_id,
                         array_ids=list(range(1, deep_config.best_n_models+1)))
    fit_id = deep_manager.dispatch_job()

    # collect job
    coll_cmd: str = f'{PYTHON_BIN} {cli_path} --config {config_path} --collect'
    watch_manager.set_job([coll_cmd], out_path, deep_config.job_config, dependency=fit_id)
    return watch_manager.dispatch_job()

def run_conv(config_path: Path, dependency: int | None = None) -> int:
    """
    Run conversion.

    Args:
        - config_path: the path to the configuration file.
        - dependency: the job dependency.

    Returns:
        int: The id of the last watcher job.
    """
    gen_config = ConfigReader(config_path).get_general_config()
    cli_path: Path = Path(__file__).resolve().parent / 'run_conv.py'
    conv_cmd: str = f'{PYTHON_BIN} {cli_path} --config {config_path}'
    conv_manager = DispatcherManager(JobType.CONV.value, gen_config.model_name, gen_config.cluster)
    conv_manager.set_job([conv_cmd], gen_config.sweep_path, gen_config.job_config, dependency=dependency)
    return conv_manager.dispatch_job()

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
    cli_path: Path = Path(__file__).resolve().parent / 'run_inf.py'
    out_path: Path = inf_config.sweep_path / INFERENCE_BENCH_DIR_NAME
    out_path.mkdir(exist_ok=True)
    watch_manager = DispatcherManager(
        JobType.WATCH_INF.value, inf_config.model_name, inf_config.job_config.cluster)
    inf_manager = DispatcherManager(
        JobType.INF.value, inf_config.model_name, inf_config.job_config.cluster)

    # init job
    init_cmd: str = f'{PYTHON_BIN} {cli_path} --config {config_path}'
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
    cli_path: Path = Path(__file__).resolve().parent / 'run_sim.py'
    out_path: Path = sim_config.sweep_path / PROPERTIES_BENCH_DIR_NAME
    out_path.mkdir(exist_ok=True)
    watch_manager = DispatcherManager(
        JobType.WATCH_SIM.value, sim_config.model_name, sim_config.job_config.cluster)
    sim_manager = DispatcherManager(
        JobType.SIM.value, sim_config.model_name, sim_config.job_config.cluster)

    # init job
    init_cmd: str = f'{PYTHON_BIN} {cli_path} --config {config_path}'
    watch_manager.set_job([init_cmd], out_path, sim_config.job_config, dependency=dependency)
    init_id = watch_manager.dispatch_job()

    # run jobs
    n_cpu = int(sim_config.job_config.slurm_opts['cpus_per_task'])
    sim_cmd: str = ' '.join([str(cmd) for cmd in [
        'bash', SUBMIT_SCRIPT_NAME,
        f'"{sim_config.lammps_bin_path} {get_lammps_params(sim_config.model_name)}"',
        PropertiesSimulator.LAMMPS_INPS_PATH,
        PropertiesSimulator.PPS_PYTHON_PATH,
        PropertiesSimulator.REF_DATA_PATH, n_cpu
    ]])
    sim_manager.set_job([sim_cmd], out_path, sim_config.job_config, dependency=init_id,
                        array_ids=list(range(1, sim_config.best_n_models+1)))
    return sim_manager.dispatch_job()

if __name__ == '__main__':
    args: Namespace = parse_args()
    conf_path: Path = Path(args.config).resolve()
    next_id: int | None = None

    if args.nohyper:
        next_id = run_hyp(conf_path, args.hypiter)

    if args.nodeep:
        next_id = run_deep(conf_path, dependency=next_id)

    if args.noconversion:
        next_id = run_conv(conf_path, dependency=next_id)

    if args.noinference:
        run_inf(conf_path, dependency=next_id)

    if args.noproperties:
        run_sim(conf_path, dependency=next_id)
