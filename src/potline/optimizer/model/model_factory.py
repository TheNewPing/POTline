"""
XpotModel interface and factory function.
"""

from pathlib import Path

import hjson # type: ignore

from .hpc_mlp import HPCMLP
from .hpc_pace import HPCPACE

def create_xpot_model(config_path: Path) -> HPCMLP:
    with open(config_path, 'r', encoding='utf-8') as file:
        config_data: dict = hjson.load(file)
        if config_data['xpot']['fitting_executable'] == 'pacemaker':
            return HPCPACE(str(config_path))
        raise ValueError('Model not supported.')
