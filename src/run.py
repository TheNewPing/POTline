"""
CLI script for dispatching PotLine.
"""

from argparse import Namespace, ArgumentParser
from pathlib import Path

from potline.dispatcher import DispatcherManager, JobType
from potline.config_reader import ConfigReader

def parse_args() -> Namespace:
    """
    Parse the command line arguments.
    """
    parser: ArgumentParser = ArgumentParser(description='Process some parameters.')
    parser.add_argument('--config', type=str, help='Path to the config file')
    parser.add_argument('--nohyper', action='store_false', help='Disable hyperparameter search')
    parser.add_argument('--nodeep', action='store_false', help='Disable deep training')
    parser.add_argument('--noconversion', action='store_false', help='Disable yace conversion')
    parser.add_argument('--noinference', action='store_false', help='Disable inference benchmark')
    parser.add_argument('--noproperties', action='store_false', help='Disable properties simulation')
    return parser.parse_args()

if __name__ == '__main__':
    args: Namespace = parse_args()
    config_path: Path = Path(args.config).resolve()
    cli_path: Path = Path('src/').resolve()
    gen_config = ConfigReader(config_path).get_general_config()

    hyp_id: int | None = None
    if args.nohyper:
        hyp_config = ConfigReader(config_path).get_optimizer_config()
        hyp_cmd = f'python {cli_path / "run_hyp.py"} {config_path}'
        hyp_manager = DispatcherManager(JobType.WATCH_FIT.value, gen_config.model_name, gen_config.cluster)
        hyp_manager.set_job([hyp_cmd], hyp_config.sweep_path, hyp_config.job_config)
        hyp_id = hyp_manager.dispatch_job()

    deep_id: int | None = None
    if args.nodeep:
        deep_config = ConfigReader(config_path).get_deep_train_config()
        deep_cmd = f'python {cli_path / "run_deep.py"} {config_path}'
        deep_manager = DispatcherManager(JobType.WATCH_DEEP.value, gen_config.model_name, gen_config.cluster)
        deep_manager.set_job([deep_cmd], deep_config.sweep_path, deep_config.job_config,
                            dependency=hyp_id)
        deep_id = deep_manager.dispatch_job()

    conv_id: int | None = None
    if args.noconversion:
        conv_cmd = f'python {cli_path / "run_conv.py"} {config_path}'
        conv_manager = DispatcherManager(JobType.CONV.value, gen_config.model_name, gen_config.cluster)
        conv_manager.set_job([conv_cmd], gen_config.sweep_path, gen_config.job_config,
                            dependency=deep_id)
        conv_id = conv_manager.dispatch_job()

    inf_id: int | None = None
    if args.noinference:
        inf_config = ConfigReader(config_path).get_bench_config()
        inf_cmd = f'python {cli_path / "run_inf.py"} {config_path}'
        inf_manager = DispatcherManager(JobType.WATCH_INF.value, gen_config.model_name, gen_config.cluster)
        inf_manager.set_job([inf_cmd], inf_config.sweep_path, inf_config.job_config,
                            dependency=conv_id)
        inf_id = inf_manager.dispatch_job()

    sim_id: int | None = None
    if args.noproperties:
        prop_config = ConfigReader(config_path).get_prop_config()
        prop_cmd = f'python {cli_path / "run_sim.py"} {config_path}'
        prop_manager = DispatcherManager(JobType.WATCH_SIM.value, gen_config.model_name, gen_config.cluster)
        prop_manager.set_job([prop_cmd], prop_config.sweep_path, prop_config.job_config,
                            dependency=conv_id)
        sim_id = prop_manager.dispatch_job()
