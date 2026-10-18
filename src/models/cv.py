"""Time-series walk-forward cross-validation splitter (Phase 5).

Generates expanding-window chronological folds over panel data:
    max(train[date_col]) < min(val[date_col])
ensuring walk-forward validation without future-looking data leakage.
"""

from __future__ import annotations

from dataclasses import dataclass
import pandas as pd


@dataclass(frozen=True)
class TimeSeriesFold:
    """A single chronological train/val fold for walk-forward validation."""
    fold_idx: int
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    train_end_date: pd.Timestamp
    val_end_date: pd.Timestamp


def generate_walk_forward_folds(
    df: pd.DataFrame,
    n_splits: int = 3,
    min_train_frac: float = 0.50,
    date_col: str = "target_date",
) -> list[TimeSeriesFold]:
    """Generate expanding-window walk-forward validation folds.

    Parameters
    ----------
    df:
        Feature DataFrame containing ``date_col``.
    n_splits:
        Number of walk-forward validation folds (default 3).
    min_train_frac:
        Fraction of unique dates allocated to the initial training window.
    date_col:
        Timestamp column defining chronological ordering.

    Returns
    -------
    list[TimeSeriesFold]
        Folds where each fold strictly satisfies:
            train_df[date_col].max() < val_df[date_col].min()
    """
    if df.empty:
        raise ValueError("Cannot generate folds on an empty DataFrame")

    unique_dates = pd.Series(sorted(df[date_col].unique()))
    n_dates = len(unique_dates)
    if n_dates < 6:
        raise ValueError(f"Insufficient unique dates for walk-forward CV: {n_dates} (minimum 6)")

    initial_train_size = int(n_dates * min_train_frac)
    if initial_train_size < 2:
        initial_train_size = 2

    remaining_dates = n_dates - initial_train_size
    if remaining_dates < n_splits:
        raise ValueError(
            f"Not enough remaining dates ({remaining_dates}) for {n_splits} splits "
            f"with initial_train_frac={min_train_frac}"
        )

    val_size = remaining_dates // n_splits
    folds: list[TimeSeriesFold] = []

    for k in range(n_splits):
        train_end_idx = initial_train_size + k * val_size
        val_end_idx = train_end_idx + val_size if k < n_splits - 1 else n_dates

        train_end_date = unique_dates.iloc[train_end_idx - 1]
        val_start_date = unique_dates.iloc[train_end_idx]
        val_end_date = unique_dates.iloc[val_end_idx - 1]

        train_df = df[df[date_col] <= train_end_date].copy().reset_index(drop=True)
        val_df = (
            df[(df[date_col] >= val_start_date) & (df[date_col] <= val_end_date)]
            .copy()
            .reset_index(drop=True)
        )

        # Leakage assertion
        assert train_df[date_col].max() < val_df[date_col].min(), (
            f"Temporal data leakage detected in fold {k}: "
            f"train max {train_df[date_col].max()} >= val min {val_df[date_col].min()}"
        )

        folds.append(
            TimeSeriesFold(
                fold_idx=k,
                train_df=train_df,
                val_df=val_df,
                train_end_date=train_end_date,
                val_end_date=val_end_date,
            )
        )

    return folds
