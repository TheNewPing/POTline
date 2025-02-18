"""
XpotModel interface and factory function.
"""

from pathlib import Path
from .model import PotModel
from ..dispatcher.slurm_preset import SupportedModel

def create_model(model_name: str, out_path: Path, pretrained: bool = False) -> PotModel:
    """
    Create a model.

    Args:
        - model_name: name of the model
        - config_filepath: path to the configuration file
        - out_path: path to the output directory
        - pretrained: flag for pretrained model
    """
    if model_name == SupportedModel.PACE.value:
        from .pace import PotPACE
        return PotPACE(out_path, pretrained)
    if model_name == SupportedModel.MACE.value:
        from .mace import PotMACE
        return PotMACE(out_path, pretrained)
    if model_name == SupportedModel.GRACE.value:
        from .grace import PotGRACE
        return PotGRACE(out_path, pretrained)

    raise ValueError(f"Unsupported model: {model_name}")

def get_fit_cmd(model_name: str, deep: bool) -> str:
    """
    Get the fitting command for a model

    Args:
        - model_name: name of the model
        - deep: flag for deep training
    """
    if model_name == SupportedModel.PACE.value:
        from .pace import PotPACE
        return PotPACE.get_fit_cmd(deep)
    if model_name == SupportedModel.MACE.value:
        from .mace import PotMACE
        return PotMACE.get_fit_cmd(deep)
    if model_name == SupportedModel.GRACE.value:
        from .grace import PotGRACE
        return PotGRACE.get_fit_cmd(deep)

    raise ValueError(f"Unsupported model: {model_name}")

def get_lammps_params(model_name: str) -> str:
    """
    Get the LAMMPS parameters for a model

    Args:
        - model_name: name of the model
    """
    if model_name == SupportedModel.PACE.value:
        from .pace import PotPACE
        return PotPACE.get_lammps_params()
    if model_name == SupportedModel.MACE.value:
        from .mace import PotMACE
        return PotMACE.get_lammps_params()
    if model_name == SupportedModel.GRACE.value:
        from .grace import PotGRACE
        return PotGRACE.get_lammps_params()

    raise ValueError(f"Unsupported model: {model_name}")
