"""
XpotModel interface and factory function.
"""

from pathlib import Path
from .model import PotModel
from ..dispatcher.slurm_preset import SupportedModel

def create_model(model_name: str,
                 config_filepath: Path,
                 out_path: Path) -> PotModel:
    """
    Create a model.

    Args:
        - model_name: name of the model
        - config_filepath: path to the configuration file
        - out_path: path to the output directory
    """
    if model_name == SupportedModel.PACE.value:
        from .pace import PotPACE
        return PotPACE(config_filepath, out_path)
    elif model_name == SupportedModel.MACE.value:
        from .mace import PotMACE
        return PotMACE(config_filepath, out_path)
    elif model_name == SupportedModel.GRACE.value:
        from .grace import PotGRACE
        return PotGRACE(config_filepath, out_path)

    raise ValueError(f"Unsupported model: {model_name}")

def create_model_from_path(model_name: str,
                           out_path: Path) -> PotModel:
    """
    Create a model from a path. Thee model must be already trained.

    Args:
        - model_name: name of the model
        - out_path: path to the output directory
    """
    if model_name == SupportedModel.PACE.value:
        from .pace import PotPACE
        return PotPACE.from_path(out_path)
    elif model_name == SupportedModel.MACE.value:
        from .mace import PotMACE
        return PotMACE.from_path(out_path)
    elif model_name == SupportedModel.GRACE.value:
        from .grace import PotGRACE
        return PotGRACE.from_path(out_path)

    raise ValueError(f"Unsupported model: {model_name}")
