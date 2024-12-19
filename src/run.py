"""
CLI script for dispatching PotLine.
"""

from argparse import Namespace, ArgumentParser
from pathlib import Path

from potline.dispatcher import DispatcherManager, JobType
from potline.config_reader import ConfigReader

def parse_args() -> Namespace:
    """
    Parse the command line arguments.
    """
    parser: ArgumentParser = ArgumentParser(description='Process some parameters.')
    parser.add_argument('--config', type=str, help='Path to the config file')
    parser.add_argument('--nohyper', action='store_false', help='Disable hyperparameter search')
    parser.add_argument('--nodeep', action='store_false', help='Disable deep training')
    parser.add_argument('--noconversion', action='store_false', help='Disable yace conversion')
    parser.add_argument('--noinference', action='store_false', help='Disable inference benchmark')
    parser.add_argument('--noproperties', action='store_false', help='Disable properties simulation')
    return parser.parse_args()

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

    disp_manager = DispatcherManager(JobType.MAIN.value, config.model_name, config.cluster)
    disp_manager.set_job(commands, config.sweep_path, config.job_config)
    disp_manager.dispatch_job()
