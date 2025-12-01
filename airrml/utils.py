"""
Utility helpers for reproducibility, logging, and simple I/O tasks.
"""
import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Optional

import numpy as np


def seed_everything(seed: int = 42) -> None:
    """
    Set seeds across random, numpy, and torch (if installed) for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        # Torch is optional for classical models; ignore if unavailable.
        pass


def get_logger(name: str = "airrml", level: int = logging.INFO) -> logging.Logger:
    """
    Configure and return a module-level logger.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def ensure_dir(path: Path) -> None:
    """
    Create a directory path if it does not already exist.
    """
    path.mkdir(parents=True, exist_ok=True)


def save_json(obj: Any, path: Path) -> None:
    """
    Persist a JSON-serializable object to disk.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2)


def load_json(path: Path) -> Optional[Any]:
    """
    Load a JSON object from disk if it exists.
    """
    if not path.exists():
        return None
    with path.open("r") as f:
        return json.load(f)
