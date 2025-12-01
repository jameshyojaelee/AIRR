"""
Sequence-level importance computation utilities.

Models should return scores indicating how strongly individual sequences
support the positive class so they can be surfaced in the submission file.
"""
from typing import Iterable, Optional

import pandas as pd

from airrml import config
from airrml.models.base import BaseRepertoireModel


def score_sequences(
    model: BaseRepertoireModel,
    sequences_df: pd.DataFrame,
    sequence_col: str = "junction_aa",
) -> pd.DataFrame:
    """
    Use a trained model to assign importance scores to sequences.
    """
    # TODO: delegate to model.get_sequence_importance or implement model-agnostic scoring
    raise NotImplementedError


def select_top_sequences(
    scored_sequences: pd.DataFrame,
    top_k: int,
    score_col: str = "importance_score",
) -> pd.DataFrame:
    """
    Select the top-k sequences by importance for submission.
    """
    # TODO: implement sorting, deduplication, and optional per-repertoire caps
    raise NotImplementedError


def format_sequence_rows(
    sequences_df: pd.DataFrame,
    dataset_name: str,
    start_id: int = 1,
    probability_placeholder: float = config.PROBABILITY_PLACEHOLDER,
) -> pd.DataFrame:
    """
    Format scored sequences into submission-ready rows.
    """
    # TODO: attach submission IDs and fill placeholder probability values
    raise NotImplementedError
