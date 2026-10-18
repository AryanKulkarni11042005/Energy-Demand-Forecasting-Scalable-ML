"""Phase 8/13 - Airflow DAG: automated ingestion + retraining pipeline.

Runs daily on schedule. Compatible with both Airflow 2.10+ and Airflow 3.x,
and portable across Docker containers and local developer environments.

Pipeline:
    fetch (incremental) -> validate (gate) -> train candidate -> promote gate
        -> reload live serving models

Exit codes from underlying scripts gate progression:
    fetch_raw_data.py       0 = ok, 1 = failed / no data
    validate_raw_data.py    0 = passed, 1 = structural errors, 2 = no data
    train_and_evaluate.py   0 = ok
    promote_model.py        0 = promoted, 1 = rejected (regression), 2 = error

A rejected promotion is NOT a pipeline failure: it means the guardrail
worked and the previous production model was correctly kept in place.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pendulum

# Airflow 3.x / 2.x unified imports
try:
    from airflow.sdk import dag, task
    from airflow.sdk.exceptions import AirflowSkipException
except ImportError:
    from airflow.decorators import dag, task
    from airflow.exceptions import AirflowSkipException

try:
    from airflow.providers.standard.operators.bash import BashOperator
except ImportError:
    from airflow.operators.bash import BashOperator

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", "/opt/airflow/project")
PROJECT_PYTHON = os.environ.get("PROJECT_PYTHON", "python")


def run(script_and_args: str) -> str:
    """Build a shell command that runs a project script with the configured python."""
    return f'cd "{PROJECT_ROOT}" && {PROJECT_PYTHON} {script_and_args}'


default_args = {
    "owner": "energy-forecasting",
    "retries": 1,
    "retry_delay": pendulum.duration(minutes=5),
}


@dag(
    dag_id="energy_demand_retraining_pipeline",
    description="Incremental EIA ingestion, validation, retraining, and gated promotion.",
    schedule="0 6 * * *",  # daily at 06:00
    start_date=pendulum.datetime(2026, 1, 1, tz="America/New_York"),
    catchup=False,
    default_args=default_args,
    tags=["energy", "retraining", "mlops"],
)
def energy_demand_retraining_pipeline():

    fetch = BashOperator(
        task_id="fetch_incremental_data",
        bash_command=run("scripts/fetch_raw_data.py --incremental"),
    )

    validate = BashOperator(
        task_id="validate_raw_data",
        bash_command=run("scripts/validate_raw_data.py"),
    )

    generate_drift_report = BashOperator(
        task_id="generate_drift_report",
        bash_command=run(
            "scripts/run_drift_monitoring.py "
            f'--output-html "{PROJECT_ROOT}/data/monitoring/drift_report.html" '
            f'--output-json "{PROJECT_ROOT}/data/monitoring/drift_metrics.json"'
        ),
    )

    train_candidate = BashOperator(
        task_id="train_candidate_model",
        bash_command=run(
            "scripts/train_and_evaluate.py "
            f'--models-dir "{PROJECT_ROOT}/models_candidate" '
            f'--output-summary "{PROJECT_ROOT}/data/evaluation_summary_candidate.json"'
        ),
    )

    @task(task_id="promote_if_not_regressed")
    def promote_if_not_regressed() -> str:
        result = subprocess.run(
            [
                PROJECT_PYTHON,
                f"{PROJECT_ROOT}/scripts/promote_model.py",
                "--candidate-models-dir",
                f"{PROJECT_ROOT}/models_candidate",
                "--candidate-summary",
                f"{PROJECT_ROOT}/data/evaluation_summary_candidate.json",
                "--live-models-dir",
                f"{PROJECT_ROOT}/models",
                "--live-summary",
                f"{PROJECT_ROOT}/data/evaluation_summary.json",
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )
        print(result.stdout)
        print(result.stderr)

        if result.returncode == 0:
            return "promoted"
        if result.returncode == 1:
            # Guardrail correctly rejected a regressed candidate; not a
            # pipeline failure, but downstream reload should be skipped.
            raise AirflowSkipException("Candidate rejected by promotion gate (regression detected).")
        raise RuntimeError(f"promote_model.py errored (exit {result.returncode})")

    reload_live_models = BashOperator(
        task_id="reload_live_models",
        bash_command=(
            'curl -sf -X POST http://host.docker.internal:8000/admin/reload-models '
            '|| curl -sf -X POST http://127.0.0.1:8000/admin/reload-models '
            '|| echo "serving app not reachable; models will load fresh on next server start"'
        ),
    )

    promotion = promote_if_not_regressed()

    fetch >> validate >> generate_drift_report >> train_candidate >> promotion >> reload_live_models


energy_demand_retraining_pipeline()
