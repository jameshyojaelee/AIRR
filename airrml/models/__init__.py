"""
Model registry and factory utilities.

Use the @register_model decorator to add new models that subclass
BaseRepertoireModel, and retrieve them via get_model(name, **kwargs).
"""
import importlib
import sys
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
    import_errors = []
    if name not in MODEL_REGISTRY:
        # Lazy import model modules to populate registry
        for module in ["kmer_logreg", "gradient_boosting", "deep_mil", "stacked_ensemble", "tcrdist_knn"]:
            try:
                importlib.import_module(f"airrml.models.{module}")
            except Exception as e:
                import_errors.append(f"{module}: {e}")
        if import_errors:
            print(f"Model import errors encountered: {import_errors}", file=sys.stderr)
    if name not in MODEL_REGISTRY:
        extra = f" Import errors: {import_errors}" if import_errors else ""
        raise ValueError(f"Model '{name}' is not registered. Available: {list(MODEL_REGISTRY.keys())}.{extra}")
    return MODEL_REGISTRY[name](**kwargs)


__all__ = ["BaseRepertoireModel", "register_model", "get_model", "MODEL_REGISTRY"]
