"""
CLI entry point for running inference benchmark.
"""

from argparse import Namespace
from pathlib import Path

from .utils import parse_config, get_model_trackers, filter_best_loss
from ..inference_bencher import InferenceBencher
from ..config_reader import ConfigReader
from ..dispatcher import DispatcherManager, JobType

if __name__ == '__main__':
    args: Namespace = parse_config()
    config_path: Path = Path(args.config).resolve()
    opt_config = ConfigReader(config_path).get_optimizer_config()
    gen_config = ConfigReader(config_path).get_general_config()
    inf_config = ConfigReader(config_path).get_bench_config()

    tracker_list = get_model_trackers(gen_config.sweep_path, gen_config.model_name)
    best_trackers = filter_best_loss(tracker_list, opt_config.energy_weight, gen_config.best_n_models)

    InferenceBencher(inf_config, best_trackers, DispatcherManager(
        JobType.INF.value, gen_config.model_name, inf_config.job_config.cluster)).run()
