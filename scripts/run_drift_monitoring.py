"""CLI runner for Phase 12 Evidently AI drift & performance monitoring."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.monitoring.drift import ModelMonitor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Evidently AI Drift & Performance Monitoring.")
    parser.add_argument(
        "--features-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "features.parquet",
        help="Path to processed features parquet",
    )
    parser.add_argument(
        "--output-html",
        type=Path,
        default=PROJECT_ROOT / "data" / "monitoring" / "drift_report.html",
        help="Path to output HTML report",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=PROJECT_ROOT / "data" / "monitoring" / "drift_metrics.json",
        help="Path to output JSON summary",
    )
    parser.add_argument("--reference-days", type=int, default=90)
    parser.add_argument("--current-days", type=int, default=30)
    parser.add_argument("--drift-share-threshold", type=float, default=0.30)
    parser.add_argument("--fail-on-drift", action="store_true", help="Exit 1 if dataset drift detected")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.features_path.exists():
        print(f"ERROR: Features file not found at {args.features_path}", file=sys.stderr)
        return 2

    print("\n" + "=" * 70)
    print("  EVIDENTLY AI: DATA & PERFORMANCE DRIFT MONITORING")
    print("=" * 70)

    df = pd.read_parquet(args.features_path)
    monitor = ModelMonitor(drift_share_threshold=args.drift_share_threshold)

    try:
        ref_df, curr_df = monitor.prepare_datasets(
            df=df,
            reference_days=args.reference_days,
            current_days=args.current_days,
        )
    except Exception as e:
        print(f"ERROR: Dataset splitting failed: {e}", file=sys.stderr)
        return 2

    print(f"  Reference window: {ref_df['target_date'].min().date()} -> {ref_df['target_date'].max().date()} ({len(ref_df):,} rows)")
    print(f"  Current window:   {curr_df['target_date'].min().date()} -> {curr_df['target_date'].max().date()} ({len(curr_df):,} rows)")

    report, summary = monitor.build_report(reference_df=ref_df, current_df=curr_df)
    monitor.save_reports(report, summary, args.output_html, args.output_json)

    print("-" * 70)
    print(f"  Total features evaluated:  {summary.total_features_evaluated}")
    print(f"  Features drifted:          {summary.drifted_features_count} ({summary.share_of_drifted_features * 100:.1f}%)")
    print(f"  Dataset drift status:      {'DRIFT DETECTED' if summary.dataset_drift_detected else 'STABLE'}")
    print(f"  Target drift status:       {'DRIFT DETECTED' if summary.target_drift_detected else 'STABLE'} (score={summary.target_drift_score})")
    print("=" * 70)

    if summary.dataset_drift_detected and args.fail_on_drift:
        print("  RESULT: Alert triggered - dataset drift exceeds threshold.\n")
        return 1

    print("  RESULT: OK - monitoring completed successfully.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
