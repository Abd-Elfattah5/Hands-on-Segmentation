"""Component registry and model builder from configuration."""

import importlib
from collections.abc import Callable
from typing import Any, TypeVar

from src.models.base import BaseSegmentationModel

MODEL_REGISTRY: dict[str, type[BaseSegmentationModel]] = {}
M = TypeVar("M", bound=type[BaseSegmentationModel])

_DEFAULT_MODULES = ("src.models.unet",)


def register_model(name: str) -> Callable[[M], M]:
    """Decorator to register a segmentation model class."""
    def decorator(cls: M) -> M:
        if not isinstance(cls, type) or not issubclass(cls, BaseSegmentationModel):
            raise TypeError(f"Model '{name}' must subclass BaseSegmentationModel, got {cls!r}")
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator


def _ensure_default_components() -> None:
    for module in _DEFAULT_MODULES:
        try:
            importlib.import_module(module)
        except ModuleNotFoundError as exc:
            if exc.name != module:
                raise


def get_model(name: str) -> type[BaseSegmentationModel]:
    if name not in MODEL_REGISTRY:
        _ensure_default_components()
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Model '{name}' not found. Available models: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name]


def build_model_from_config(config: dict[str, Any]) -> BaseSegmentationModel:
    """Build and return a model instance configured per ``config['model']``."""
    model_cfg = dict(config.get("model", {}))
    name = model_cfg.pop("name", "resnet34_unet")
    model_cls = get_model(name)
    model = model_cls(**model_cfg)
    model.config = config
    return model
