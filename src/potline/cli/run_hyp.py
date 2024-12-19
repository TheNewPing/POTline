"""
CLI entry point for running hyperparameter search.
"""

from argparse import Namespace
from pathlib import Path

from .utils import parse_config
from ..hyper_searcher import PotOptimizer
from ..config_reader import ConfigReader
from ..dispatcher import DispatcherManager, JobType

if __name__ == '__main__':
    args: Namespace = parse_config()
    config_path: Path = Path(args.config).resolve()
    config = ConfigReader(config_path).get_optimizer_config()

    PotOptimizer(config, DispatcherManager(
        JobType.FIT.value, config.model_name, config.job_config.cluster)).run()
