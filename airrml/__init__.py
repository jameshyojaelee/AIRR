"""
AIRR-ML-25 package skeleton.

This module exposes configuration and model registry helpers without
forcing heavyweight imports at package import time.
"""
from . import config
from .models import get_model, register_model

__all__ = ["config", "get_model", "register_model"]
