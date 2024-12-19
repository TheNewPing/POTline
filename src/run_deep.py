"""
CLI entry point for running deep training.
"""

from argparse import Namespace
from pathlib import Path

from potline.utils import parse_config, get_model_trackers, filter_best_loss
from potline.deep_trainer import DeepTrainer
from potline.config_reader import ConfigReader
from potline.dispatcher import DispatcherManager, JobType

if __name__ == '__main__':
    args: Namespace = parse_config()
    config_path: Path = Path(args.config).resolve()
    opt_config = ConfigReader(config_path).get_optimizer_config()
    gen_config = ConfigReader(config_path).get_general_config()
    deep_config = ConfigReader(config_path).get_deep_train_config()

    tracker_list = get_model_trackers(gen_config.sweep_path, gen_config.model_name)
    best_trackers = filter_best_loss(tracker_list, opt_config.energy_weight, gen_config.best_n_models)

    DeepTrainer(deep_config, best_trackers, DispatcherManager(
        JobType.DEEP.value, gen_config.model_name, deep_config.job_config.cluster)).run()
