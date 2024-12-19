"""
CLI entry point for running hyperparameter search.
"""

from argparse import Namespace
from pathlib import Path

from potline.utils import parse_config
from potline.hyper_searcher import PotOptimizer
from potline.config_reader import ConfigReader
from potline.dispatcher import DispatcherManager, JobType

if __name__ == '__main__':
    args: Namespace = parse_config()
    config_path: Path = Path(args.config).resolve()
    config = ConfigReader(config_path).get_optimizer_config()

    PotOptimizer(config, DispatcherManager(
        JobType.FIT.value, config.model_name, config.job_config.cluster)).run()
