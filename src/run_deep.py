"""
CLI entry point for running deep training.
"""

from argparse import Namespace, ArgumentParser
from pathlib import Path

from potline.utils import parse_config, get_model_trackers, filter_best_loss
from potline.deep_trainer import DeepTrainer
from potline.config_reader import ConfigReader

def parse_deep() -> Namespace:
    """
    Parse the deep training arguments.
    """
    parser: ArgumentParser = ArgumentParser(description='Process some parameters.')
    parser.add_argument('--collect', action='store_true', help='Collect losses')
    return parser.parse_args()

if __name__ == '__main__':
    args: Namespace = parse_config()
    deep_args: Namespace = parse_deep()
    config_path: Path = Path(args.config).resolve()
    deep_config = ConfigReader(config_path).get_deep_train_config()

    tracker_list = get_model_trackers(deep_config.sweep_path, deep_config.model_name,
                                      force_from_hyp=not deep_args.collect)
    best_trackers = filter_best_loss(tracker_list, deep_config.energy_weight, deep_config.best_n_models)

    if not deep_args.collect:
        DeepTrainer(config_path, best_trackers).prep_deep()
    else:
        DeepTrainer(config_path, best_trackers).collect()
