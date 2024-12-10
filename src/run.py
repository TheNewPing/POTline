"""
CLI script for running PotLine.
"""

from argparse import Namespace, ArgumentParser
from pathlib import Path

from potline.dispatcher import DispatcherFactory, JobType, SupportedModel
from potline.config_reader import ConfigReader

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

    config = ConfigReader(Path(args.config)).get_general_config()
    cmd_args = f' --config {args.config}' if args.config else '' + \
                ' --nohyper' if args.nohyper else '' + \
                ' --nodeep' if args.nodeep else '' + \
                ' --noconversion' if args.noconversion else '' + \
                ' --noinference' if args.noinference else '' + \
                ' --noproperties' if args.noproperties else ''
    commands = [f'cd {config.sweep_path}', f'python main.py {cmd_args}']

    for model in SupportedModel:
        if model.value == config.model_name:
            dispatcher = DispatcherFactory(JobType.MAIN, config.cluster).create_dispatcher(
                commands, config.sweep_path, model)
            dispatcher.dispatch()

    raise NotImplementedError("Not implemented yet.")
