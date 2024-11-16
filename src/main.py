"""
CLI script for running PotLine.
"""

from argparse import Namespace, ArgumentParser
from potline import PotLine

if __name__ == '__main__':
    def parse_args() -> Namespace:
        parser: ArgumentParser = ArgumentParser(description='Process some parameters.')
        parser.add_argument('--config', type=str, required=True, help='Path to the config file')
        parser.add_argument('--iterations', type=int, required=True, help='Number of iterations')
        parser.add_argument('--inference', action='store_true', help='Enable inference benchmark')
        parser.add_argument('--properties', action='store_true', help='Enable properties simulation')
        return parser.parse_args()

    args: Namespace = parse_args()
    config_path: str = args.config
    iterations: int = args.iterations
    inference_flag: bool = args.inference
    data_analysis_flag: bool = args.properties

    potline = PotLine(config_path, iterations, inference_flag, data_analysis_flag)
    potline.run()
