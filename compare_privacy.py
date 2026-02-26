import subprocess
import os
import sys
from pathlib import Path
import pandas as pd
import json

# Allow importing sd_eval without installing to site-packages
SD_EVAL_ROOT = "/mnt/c/Users/gaura/Documents/GitHub/sd_eval"
SD_EVAL_SRC = os.path.join(SD_EVAL_ROOT, "src")
if SD_EVAL_SRC not in sys.path:
    sys.path.insert(0, SD_EVAL_SRC)

try:
    from sd_eval.api import evaluate
except Exception as exc:  # pragma: no cover - defensive import guard
    raise RuntimeError(
        "sd_eval could not be imported. Ensure the repo exists at "
        f"{SD_EVAL_ROOT} and dependencies are installed."
    ) from exc


def run_experiment(dataname, column, value, method, num_samples=100):
    print(f"\n>>> Running: Dataset={dataname}, Condition={column}={value}, Method={method}")

    cmd = [
        "python",
        "generate_conditional.py",
        "--dataname",
        dataname,
        "--condition_column",
        column,
        "--condition_value",
        str(value),
        "--num_samples",
        str(num_samples),
        "--privacy_method",
        method,
        "--output_dir",
        f"privacy_comparison/{method}",
    ]

    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running experiment: {e}")
        return False


def _sd_eval_paths(dataname):
    real_path = os.path.join(SD_EVAL_ROOT, "tests", "static", "datasets", dataname, "train.csv")
    info_path = os.path.join(SD_EVAL_ROOT, "tests", "static", "datasets", dataname, "info.json")
    config_path = os.path.join(SD_EVAL_ROOT, "configs", "default.yaml")

    for p in [real_path, info_path, config_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Required sd_eval asset missing: {p}")

    return real_path, info_path, config_path


def _prepare_synthetic(csv_path: str) -> str:
    """Normalize column names so they match sd_eval schemas (hyphens instead of dots)."""
    df = pd.read_csv(csv_path)
    df.rename(columns=lambda c: c.replace(".", "-"), inplace=True)
    tmp_path = Path(csv_path).with_suffix("").as_posix() + "_eval.csv"
    df.to_csv(tmp_path, index=False)
    return tmp_path


def evaluate_with_sd_eval(csv_path, dataname, column, value, save_json=True):
    real_path, info_path, config_path = _sd_eval_paths(dataname)
    condition = f"{column}={str(value).strip()}"
    eval_ready_path = _prepare_synthetic(csv_path)
    report = evaluate(
        real_path=real_path,
        synthetic_path=eval_ready_path,
        info_path=info_path,
        condition=condition,
        config_path=config_path,
    )
    if not isinstance(report, dict):
        raise TypeError(f"sd_eval returned {type(report)} instead of dict")

    # Optionally persist full report JSON next to the CSV
    if save_json:
        json_path = Path(csv_path).with_suffix("").as_posix() + "_sd_eval.json"
        with open(json_path, "w") as f:
            json.dump(report, f, indent=2)

    metrics = report.get("metrics") or {}

    def _score(payload):
        if isinstance(payload, dict):
            return payload.get("score")
        if isinstance(payload, (float, int)):
            return float(payload)
        return None

    scores = {name: _score(payload) for name, payload in metrics.items()}

    composite_payload = report.get("composite")
    scores["composite"] = (
        composite_payload.get("score") if isinstance(composite_payload, dict) else composite_payload
    )

    return scores


def _real_row_count(dataname: str) -> int:
    real_path, _, _ = _sd_eval_paths(dataname)
    return len(pd.read_csv(real_path))


def compare_results(experiments):
    results = []
    metric_columns: set[str] = set()

    for dataname, column, value in experiments:
        for method in ["none", "stochastic", "midpoint"]:
            file_path = f"privacy_comparison/{method}/{dataname}_{column}_{value}.csv"
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                # Simple stats
                row = {
                    "dataset": dataname,
                    "condition": f"{column}={value}",
                    "method": method,
                    "num_samples": len(df),
                    "unique_values_in_condition": df[column].nunique(),
                    "most_frequent_condition": df[column].mode()[0] if not df[column].empty else "N/A",
                }
                try:
                    eval_metrics = evaluate_with_sd_eval(file_path, dataname, column, value)
                    row.update(eval_metrics)
                    metric_columns.update(eval_metrics.keys())
                except Exception as exc:
                    print(f"sd_eval failed for {file_path}: {exc}")
                results.append(row)

    # Ensure all rows have all metric columns (fill missing with NaN)
    if results and metric_columns:
        for row in results:
            for m in metric_columns:
                row.setdefault(m, float("nan"))

    return pd.DataFrame(results)


if __name__ == "__main__":
    # Define experiments based on available datasets and models
    # Note: We assume models for these datasets are already trained.
    experiments = [
        ("adult", "education", " 11th"),
        ("adult", "education", " Bachelors"),
        ("adult", "occupation", " Sales"),
        ("adult", "marital-status", " Married-AF-spouse"),
        ("adult", "native-country", " Yugoslavia"),
        ("adult", "native-country", " Holand-Netherlands"),
        ("adult", "workclass", " Never-worked"),
        ("adult", "workclass", " Without-pay"),
    ]

    # Check what else is available in data/ and tabdiff/ckpt/
    # If other models are trained, they can be added here.

    os.makedirs("privacy_comparison", exist_ok=True)

    for dataname, column, value in experiments:
        target_rows = _real_row_count(dataname)
        for method in ["none", "stochastic", "midpoint"]:
            run_experiment(dataname, column, value, method, num_samples=target_rows)

    summary_df = compare_results(experiments)
    print("\n--- Privacy Comparison Summary ---")
    print(summary_df.to_string(index=False))

    summary_df.to_csv("privacy_comparison/summary.csv", index=False)
    print(f"\nSummary saved to privacy_comparison/summary.csv")
