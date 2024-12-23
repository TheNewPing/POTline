"""
CLI entry point for running hyperparameter search.
"""

from argparse import Namespace, ArgumentParser
from pathlib import Path

from potline.utils import parse_config
from potline.hyper_searcher import PotOptimizer

def parse_hyp() -> Namespace:
    """
    Parse the hyperparameter search arguments.
    """
    parser: ArgumentParser = ArgumentParser(description='Process some parameters.')
    parser.add_argument('--restart', action='store_true', help='Restart the optimizer')
    parser.add_argument('--iteration', type=int, default=1, help='Iteration number')
    return parser.parse_args()

if __name__ == '__main__':
    args: Namespace = parse_config()
    hyp_args: Namespace = parse_hyp()
    config_path: Path = Path(args.config).resolve()

    PotOptimizer(config_path, hyp_args.restart, hyp_args.iteration).run()
