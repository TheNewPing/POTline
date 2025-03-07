"""
CLI entry point for running properties simulations.
"""

from argparse import Namespace, ArgumentParser
from pathlib import Path

from potline.utils import get_model_trackers, filter_best_loss
from potline.config_reader import ConfigReader
from potline.experiment import Experiment

def parse_config() -> Namespace:
    """
    Parse the command line arguments.
    """
    parser: ArgumentParser = ArgumentParser(description='Process some parameters.')
    parser.add_argument('--config', type=str, help='Path to the config file')
    parser.add_argument('--copydir', type=str, help='Path to the directory to copy')
    parser.add_argument('--outpath', type=str, help='Path to the output directory')
    return parser.parse_args()

if __name__ == '__main__':
    args: Namespace = parse_config()
    config_path: Path = Path(args.config).resolve()
    gen_config = ConfigReader(config_path).get_general_config()

    # placeholder for pretrained models,
    # the value does not matter since the list has only 1 model
    energy_weight = 0.5
    if not gen_config.pretrained_path:
        energy_weight = ConfigReader(config_path).get_optimizer_config().energy_weight

    tracker_list = get_model_trackers(gen_config.sweep_path, gen_config.model_name,
                                      pretrained_path=gen_config.pretrained_path)
    best_trackers = filter_best_loss(tracker_list, energy_weight, gen_config.best_n_models)

    Experiment.prep_exp(Path(args.outpath), Path(args.copydir), best_trackers)
