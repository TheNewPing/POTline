"""
CLI script for running PotLine.
"""

from argparse import Namespace, ArgumentParser
from pathlib import Path

from potline import PotLine

if __name__ == '__main__':
    def parse_args() -> Namespace:
        """
        Parse the command line arguments.
        """
        parser: ArgumentParser = ArgumentParser(description='Process some parameters.')
        parser.add_argument('--config', type=str, default='src/data/config.hjson',
                            help='Path to the config file')
        parser.add_argument('--nohyper', action='store_false', help='Disable hyperparameter search')
        parser.add_argument('--nodeep', action='store_false', help='Disable deep training')
        parser.add_argument('--noconversion', action='store_false', help='Disable yace conversion')
        parser.add_argument('--noinference', action='store_false', help='Disable inference benchmark')
        parser.add_argument('--noproperties', action='store_false', help='Disable properties simulation')
        return parser.parse_args()

    args: Namespace = parse_args()

    potline = PotLine(Path(args.config),
                      args.nohyper,
                      args.nodeep,
                      args.noconversion,
                      args.noinference,
                      args.noproperties)

    print('Starting PotLine pipeline...')
    potline.run()
    print('PotLine pipeline finished.')
