"""
CLI utilities for PotLine.
"""

from pathlib import Path
from argparse import Namespace, ArgumentParser

from potline.loss_logger import ModelTracker
from potline.hyper_searcher import OPTIM_DIR_NAME
from potline.deep_trainer import DEEP_TRAIN_DIR_NAME

def parse_config() -> Namespace:
    """
    Parse the command line arguments.
    """
    parser: ArgumentParser = ArgumentParser(description='Process some parameters.')
    parser.add_argument('config', type=str, help='Path to the config file')
    return parser.parse_args()

def filter_best_loss(model_list: list[ModelTracker], energy_weight: float, n: int) -> list[ModelTracker]:
    sorted_models = sorted(model_list,
                        key=lambda model: model.get_total_valid_loss(energy_weight))
    return sorted_models[:n]

def get_model_out_paths(sweep_path: Path) -> list[Path]:
    """
    Get the paths to the models.
    """
    job_dirs: list[Path] = [d for d in sweep_path.iterdir() if d.is_dir()]
    deep_trained: bool = DEEP_TRAIN_DIR_NAME in [d.name for d in job_dirs]
    hyper_searched: bool = OPTIM_DIR_NAME in [d.name for d in job_dirs]
    if deep_trained and hyper_searched:
        latest_dir: Path = sweep_path / DEEP_TRAIN_DIR_NAME
        return [d for d in latest_dir.iterdir() if d.is_dir()]
    if hyper_searched:
        latest_dir = sweep_path / OPTIM_DIR_NAME
        iter_dirs: list[Path] = [d for d in latest_dir.iterdir() if d.is_dir()]
        return [d for d in iter_dirs for d in d.iterdir() if d.is_dir()]

    raise ValueError('No model directories found.')

def get_model_trackers(sweep_path: Path, model_name: str) -> list[ModelTracker]:
    """
    Get the model trackers from the sweep path.
    """
    models: list[ModelTracker] = []
    for model in get_model_out_paths(sweep_path):
        if model.is_dir():
            model_tracker = ModelTracker.from_path(model_name, model, sweep_path)
            models.append(model_tracker)
    return models
