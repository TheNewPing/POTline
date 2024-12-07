"""
Model utilities.
"""

import subprocess
from pathlib import Path

from .path_utils import unpatify, gen_from_template

YACE_NAME: str = 'pace.yace'
LAST_POTENTIAL_NAME: str = 'output_potential.yaml'
POTENTIAL_NAME: str = 'potential.in'
POTENTIAL_TEMPLATE_PATH: Path = Path(__file__).parent / 'template' / POTENTIAL_NAME

def run_yacer(model_name: str, pot_path: Path, out_path: Path) -> Path:
    """
    Convert the model YAML to YACE format.

    Args:
        model_name (str): The name of the model.
        pot_path (Path): The path to the potential.
        out_path (Path): The path to the output directory.

    Returns:
        Path: The path to the YACE file.
    """
    if model_name == 'pacemaker':
        subprocess.run(['pace_yaml2yace', '-o', str(out_path), str(pot_path)], check=True)
    else:
        raise ValueError(f'Unknown model name: {model_name}')
    return out_path

def convert_yace(model_name: str, sweep_path: Path) -> list[Path]:
    """
    Converts the best potentials to YACE format.

    Args:
        sweep_path (Path): The path to the sweep directory.

    Returns:
        list[Path]: List of paths to the YACE files.
    """
    # Convert the best potentials to YACE format
    yace_list: list[Path] = []
    model_dirs: list[Path] = [d for iter_dir in sweep_path.iterdir() if iter_dir.is_dir()
                              for d in iter_dir.iterdir() if d.is_dir() ]
    for model_dir in model_dirs:
        # Convert the best cycle to YACE format
        yace_list.append(run_yacer(
            model_name,
            model_dir.resolve() / LAST_POTENTIAL_NAME,
            model_dir.resolve() / YACE_NAME))
    return yace_list

def create_potential(model_name: str, yace_path: Path, out_path: Path) -> Path:
    """
    Create the potential in YACE format.

    Args:
        model_name (str): The name of the model.
        yace_path (Path): The path to the YACE file.
        out_path (Path): The path to the output directory.

    Returns:
        Path: The path to the potential.
    """
    if model_name == 'pacemaker':
        pstyle = 'pace'
    else:
        raise ValueError(f'Unknown model name: {model_name}')

    potential_values: dict = unpatify({
        'pstyle': pstyle,
        'yace_path': yace_path,
    })
    potential_file = out_path / POTENTIAL_NAME
    gen_from_template(POTENTIAL_TEMPLATE_PATH, potential_values, potential_file)

    return out_path
