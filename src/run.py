"""
CLI script for running PotLine.
"""

from argparse import Namespace
from pathlib import Path

from potline.dispatcher import DispatcherFactory, JobType
from potline.config_reader import ConfigReader

from main import parse_args

if __name__ == '__main__':
    args: Namespace = parse_args()
    config_path: Path = Path(args.config).resolve()
    main_path: Path = Path('src/main.py').resolve()

    config = ConfigReader(config_path).get_general_config()
    cmd_args = (f' --config {config_path}' if args.config else '') + \
               (' --nohyper' if not args.nohyper else '') + \
               (' --nodeep' if not args.nodeep else '') + \
               (' --noconversion' if not args.noconversion else '') + \
               (' --noinference' if not args.noinference else '') + \
               (' --noproperties' if not args.noproperties else '')
    commands = [f'cd {config.sweep_path.resolve()}',
                f'python {main_path} {cmd_args}',]

    dispatcher = DispatcherFactory(JobType.MAIN.value, config.cluster).create_dispatcher(
        commands, config.sweep_path, config.model_name)
    dispatcher.dispatch()
