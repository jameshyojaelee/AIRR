"""
Model registry and factory utilities.

Use the @register_model decorator to add new models that subclass
BaseRepertoireModel, and retrieve them via get_model(name, **kwargs).
"""
from typing import Callable, Dict, Type

from airrml.models.base import BaseRepertoireModel

MODEL_REGISTRY: Dict[str, Type[BaseRepertoireModel]] = {}


def register_model(name: str) -> Callable[[Type[BaseRepertoireModel]], Type[BaseRepertoireModel]]:
    """
    Decorator to register a model class under a given name.
    """

    def decorator(cls: Type[BaseRepertoireModel]) -> Type[BaseRepertoireModel]:
        MODEL_REGISTRY[name] = cls
        return cls

    return decorator


def get_model(name: str, **kwargs) -> BaseRepertoireModel:
    """
    Instantiate a registered model by name.
    """
    if name not in MODEL_REGISTRY:
        # Lazy import model modules to populate registry
        try:
            from airrml.models import kmer_logreg  # noqa: F401
            from airrml.models import gradient_boosting  # noqa: F401
            from airrml.models import deep_mil  # noqa: F401
        except Exception:
            pass
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{name}' is not registered. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name](**kwargs)


__all__ = ["BaseRepertoireModel", "register_model", "get_model", "MODEL_REGISTRY"]
