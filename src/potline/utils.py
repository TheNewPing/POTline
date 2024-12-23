"""
CLI utilities for PotLine.
"""

from pathlib import Path
from argparse import Namespace, ArgumentParser

from .loss_logger import ModelTracker
from .hyper_searcher import PotOptimizer
from .deep_trainer import DeepTrainer

def filter_best_loss(model_list: list[ModelTracker], energy_weight: float, n: int) -> list[ModelTracker]:
    sorted_models = sorted(model_list,
                        key=lambda model: model.get_total_valid_loss(energy_weight))
    return sorted_models[:n]


def get_model_trackers(sweep_path: Path, model_name: str,
                       force_from_hyp: bool = False) -> list[ModelTracker]:
    """
    Get the model trackers from the sweep path.

    Args:
        - sweep_path: path to the sweep
        - model_name: name of the model
        - force_from_hyp: force the model to be from hyperparameter search

    Returns:
        - list of model trackers
    """
    if force_from_hyp:
        return PotOptimizer.get_model_trackers(sweep_path, model_name)

    try:
        return DeepTrainer.get_model_trackers(sweep_path, model_name)
    except FileNotFoundError:
        return PotOptimizer.get_model_trackers(sweep_path, model_name)
