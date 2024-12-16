"""
CLI script for running PotLine.
"""

from argparse import Namespace
from pathlib import Path

from potline import PotLine

from run import parse_args

if __name__ == '__main__':
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
