"""
CLI script for dispatching PotLine.
"""

from argparse import Namespace, ArgumentParser
from pathlib import Path

from potline.config_reader import ConfigReader
from potline.model import PotModel
from potline.hyper_searcher import PotOptimizer
from potline.deep_trainer import DeepTrainer
from potline.experiment import PropertiesSimulator, InferenceBencher

def parse_args() -> Namespace:
    """
    Parse the command line arguments.
    """
    parser: ArgumentParser = ArgumentParser(description='Process some parameters.')
    parser.add_argument('--config', type=str, help='Path to the config file')
    parser.add_argument('--nohyper', action='store_false', help='Disable hyperparameter search')
    parser.add_argument('--hypiter', type=int, default=1, help='Hyperparameter search starting iteration')
    parser.add_argument('--nodeep', action='store_false', help='Disable deep training')
    parser.add_argument('--noconversion', action='store_false', help='Disable yace conversion')
    parser.add_argument('--noinference', action='store_false', help='Disable inference benchmark')
    parser.add_argument('--noproperties', action='store_false', help='Disable properties simulation')
    return parser.parse_args()

if __name__ == '__main__':
    args: Namespace = parse_args()
    conf_path: Path = Path(args.config).resolve()
    next_id: int | None = None
    gen_conf = ConfigReader(conf_path).get_general_config()
    gen_conf.sweep_path.mkdir(exist_ok=True)

    if args.nohyper:
        next_id = PotOptimizer.run_hyp(conf_path, args.hypiter)

    if args.nodeep:
        next_id = DeepTrainer.run_deep(conf_path, dependency=next_id)

    if args.noconversion:
        next_id = PotModel.run_conv(conf_path, dependency=next_id)

    if args.noinference:
        InferenceBencher(conf_path).run_inf(dependency=next_id)

    if args.noproperties:
        PropertiesSimulator(conf_path).run_sim(dependency=next_id)
