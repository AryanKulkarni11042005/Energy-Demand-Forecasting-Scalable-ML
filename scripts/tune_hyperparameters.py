"""CLI Script: Hyperparameter Tuning via Optuna (Phase 5).

Optimizes LightGBM parameters for each tier (Large, Medium, Small)
using chronological walk-forward cross-validation over the train + validation window.
The 15% test set remains completely unseen.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.validate import load_modeling_frame, load_modeling_respondents
from src.features.build_features import build_features, split_time_series
from src.models.hpo import run_tier_hpo, save_tuned_params

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("eia.hpo_cli")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune LightGBM hyperparameters per tier via Optuna.")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "eia",
        help="Path to raw EIA Parquet store",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "_logs" / "validation_report.json",
        help="Path to validation report JSON",
    )
    parser.add_argument(
        "--start",
        type=str,
        default="2022-01-01",
        help="Analysis window start date",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="Analysis window end date",
    )
    parser.add_argument(
        "--tiers",
        nargs="+",
        default=["Large", "Medium", "Small"],
        help="Tiers to tune (e.g. Large Medium Small)",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=30,
        help="Number of Optuna trials per tier",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=3,
        help="Number of walk-forward cross-validation splits",
    )
    parser.add_argument(
        "--output-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "tuned_params.json",
        help="Path to save best hyperparameters JSON",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="sqlite:///optuna.db",
        help="Optuna study storage URL (or empty/none for in-memory)",
    )
    parser.add_argument(
        "--in-memory-db",
        action="store_true",
        help="Use in-memory Optuna storage without creating optuna.db",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    respondents = None
    if args.report_path.exists():
        respondents = load_modeling_respondents(args.report_path)
        logger.info("Loaded %d approved respondents from validation report", len(respondents))

    logger.info("Loading panel data from %s...", args.raw_dir)
    panel_df = load_modeling_frame(
        args.raw_dir,
        respondents=respondents,
        start=args.start,
        end=args.end,
        wide=True,
    )

    logger.info("Building features for %d panel rows...", len(panel_df))
    features_df = build_features(panel_df, drop_na_target=True)

    # Split chronologically: train (70%), val (15%), test (15%)
    splits = split_time_series(features_df, train_frac=0.70, val_frac=0.15, test_frac=0.15)
    train_val_df = pd.concat([splits.train, splits.val], ignore_index=True)
    logger.info(
        "Tuning pool: %d rows (train + val, test set untouched) across %s -> %s",
        len(train_val_df),
        train_val_df["target_date"].min().date(),
        train_val_df["target_date"].max().date(),
    )

    storage = None if args.in_memory_db else args.db_path

    best_params_by_tier: dict[str, dict] = {}
    for tier in args.tiers:
        print(f"\n{'=' * 60}\n  TUNING HYPERPARAMETERS FOR TIER: {tier}\n{'=' * 60}")
        best_params, best_wmape = run_tier_hpo(
            tier=tier,
            train_val_df=train_val_df,
            n_trials=args.n_trials,
            n_splits=args.n_splits,
            storage_url=storage,
        )
        best_params_by_tier[tier] = best_params
        print(f"  Best Val WMAPE for {tier}: {best_wmape:.2f}%\n")

    save_tuned_params(best_params_by_tier, args.output_config)
    print(f"\nAll tuned parameters successfully written to {args.output_config}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
