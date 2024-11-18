"""
CLI script for running PotLine.
"""

from argparse import Namespace, ArgumentParser
from pathlib import Path

from potline import PotLine

if __name__ == '__main__':
    def parse_args() -> Namespace:
        parser: ArgumentParser = ArgumentParser(description='Process some parameters.')
        parser.add_argument('--config', type=str, required=True, help='Path to the config file')
        parser.add_argument('--iterations', type=int, default=1, help='Number of iterations')
        parser.add_argument('--fitting', action='store_true', help='Enable potential fitting')
        parser.add_argument('--conversion', action='store_true', help='Enable yace conversion')
        parser.add_argument('--inference', action='store_true', help='Enable inference benchmark')
        parser.add_argument('--properties', action='store_true', help='Enable properties simulation')
        parser.add_argument('--fitted', type=str, default=None, help='Path to the fitted potential')
        return parser.parse_args()

    args: Namespace = parse_args()

    potline = PotLine(Path(args.config),
                      args.iterations,
                      args.fitting,
                      args.conversion,
                      args.inference,
                      args.properties,
                      Path(args.fitted) if args.fitted else None)
    potline.run()
