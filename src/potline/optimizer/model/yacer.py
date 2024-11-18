"""
Model YAML to YACE converter.
"""

import subprocess
from pathlib import Path

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
        subprocess.run(['pace_yaml2yace', '-o', out_path, pot_path], check=True)
    return out_path

def convert_yace(model_name: str, fitted_path: Path, out_yace_path: Path) -> list[Path]:
    """
    Converts the best potentials to YACE format.

    Args:
        fitted_path (Path): The path to the sweep directory.
        out_yace_path (Path): The path to the output directory.

    Returns:
        list[Path]: List of paths to the YACE files.
    """
    # Convert the best potentials to YACE format
    yace_list: list[Path] = []
    model_dirs = [d for d in fitted_path.iterdir() if d.is_dir()]
    for model_dir in model_dirs:
        # Create the output directory
        out_dir_path = out_yace_path / model_dir
        out_dir_path.mkdir(parents=True, exist_ok=True)
        # Convert the best cycle to YACE format
        yace_list.append(run_yacer(
            model_name,
            model_dir.resolve() / 'interim_potential_best_cycle.yaml',
            out_dir_path / 'pace.yace'))
    return yace_list
