"""
XpotModel interface and factory function.
"""

from pathlib import Path

from .model import PotModel, SupportedModel
from .pace import PotPACE

_MODEL_CONSTRUCTORS: dict[SupportedModel, type[PotModel]] = {
    SupportedModel.PACE: PotPACE,
    # Add other models here as needed
}

def create_model(model_name: str,
                 config_filepath: Path,
                 out_path: Path) -> PotModel:
    """
    Create a model.
    """
    for model in SupportedModel:
        if model_name == model.value:
            model_class: type[PotModel] = _MODEL_CONSTRUCTORS[model]
            return model_class(config_filepath, out_path)

    raise ValueError(f"Unsupported model: {model_name}")

def create_model_from_path(model_name: str,
                           out_path: Path,) -> PotModel:
    """
    Create a model from a path.
    """
    for model in SupportedModel:
        if model_name == model.value:
            model_class: type[PotModel] = _MODEL_CONSTRUCTORS[model]
            return model_class.from_path(out_path)

    raise ValueError(f"Unsupported model: {model_name}")
