#!/usr/bin/env python
"""
Run a repeated conditional-generation experiment for TabDiff privacy methods.

The script compares methods such as none, stochastic, and midpoint by:
1. probing requested evaluators before the full run,
2. generating paired outputs for the same seed/condition across methods,
3. evaluating every output with working TabDiff metrics,
4. testing whether metric differences are statistically significant.

Example:
    python run_conditional_significance_experiment.py \
        --dataname adult \
        --ckpt-path tabdiff/ckpt/adult/learnable_schedule/best_ema_model.pt \
        --condition "education=11th" \
        --condition "occupation=Sales" \
        --num-samples same-as-real \
        --repeats 30 \
        --metrics density c2st dcr mle \
        --output-dir experiments/adult_tabdiff_privacy_methods
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from generate_conditional import generate_conditional_samples
from tabdiff.metrics import TabMetrics

try:
    from scipy import stats
except Exception:  # pragma: no cover - scipy is optional at runtime
    stats = None


DEFAULT_METHODS = ["none", "stochastic", "midpoint"]
DEFAULT_METRICS = ["density", "c2st", "dcr", "mle"]
DEFAULT_FRAMEWORK_EVALUATOR = "none"
MISSING_CATEGORY_TOKEN = "__MISSING__"
ID_COLUMNS = [
    "dataname",
    "condition_column",
    "condition_value",
    "condition_id",
    "seed",
    "repeat",
    "method",
    "num_samples",
    "sample_path",
    "eval_sample_path",
    "framework_report_path",
]


@dataclass(frozen=True)
class Condition:
    column: str
    value: str

    @property
    def label(self) -> str:
        return f"{self.column}={self.value}"

    @property
    def slug(self) -> str:
        return slugify(self.label)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate conditional samples with multiple TabDiff privacy methods "
            "and test metric differences with confidence intervals."
        )
    )
    parser.add_argument("--dataname", required=True, help="Dataset name under data/<dataname>.")
    parser.add_argument(
        "--condition",
        action="append",
        default=[],
        help="Condition in column=value form. Repeat this flag for multiple conditions.",
    )
    parser.add_argument(
        "--conditions-json",
        default=None,
        help="Optional JSON file containing [{'column': ..., 'value': ...}, ...].",
    )
    parser.add_argument("--ckpt-path", default=None, help="Checkpoint path for generate_conditional.py.")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=DEFAULT_METHODS,
        choices=DEFAULT_METHODS,
        help="Methods to compare.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=DEFAULT_METRICS,
        choices=DEFAULT_METRICS,
        help="TabDiff metrics to try. Broken metrics are skipped unless --strict-metrics is set.",
    )
    parser.add_argument(
        "--num-samples",
        default="same-as-real",
        help="Samples per method/repeat, or 'same-as-real'.",
    )
    parser.add_argument("--repeats", type=int, default=30, help="Number of paired repeats/seeds.")
    parser.add_argument("--seed-start", type=int, default=0, help="First seed.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Sampling/eval device.")
    parser.add_argument("--w-num", type=float, default=0.0, help="Numerical guidance weight.")
    parser.add_argument("--w-cat", type=float, default=0.0, help="Categorical guidance weight.")
    parser.add_argument("--resample-rounds", type=int, default=1, help="Imputation resampling rounds.")
    parser.add_argument(
        "--stochastic-start-ratio",
        type=float,
        default=None,
        help=(
            "Optional override for stochastic_start_ratio passed to generate_conditional.py. "
            "When omitted, generate_conditional.py defaults are used."
        ),
    )
    parser.add_argument(
        "--stochastic-s-churn",
        type=float,
        default=None,
        help=(
            "Optional override for s_churn in the stochastic method. "
            "When omitted, generate_conditional.py defaults are used."
        ),
    )
    parser.add_argument(
        "--stochastic-cat-noise-scale",
        type=float,
        default=None,
        help=(
            "Optional override for cat_noise_scale in the stochastic method. "
            "When omitted, generate_conditional.py defaults are used."
        ),
    )
    parser.add_argument(
        "--midpoint-privacy-noise-scale",
        type=float,
        default=None,
        help=(
            "Optional override for privacy_noise_scale in the midpoint method. "
            "When omitted, generate_conditional.py defaults are used."
        ),
    )
    parser.add_argument(
        "--midpoint-cat-noise-scale",
        type=float,
        default=None,
        help=(
            "Optional override for cat_noise_scale in the midpoint method. "
            "When omitted, generate_conditional.py defaults are used."
        ),
    )
    parser.add_argument("--real-data-path", default=None, help="Override real CSV path.")
    parser.add_argument("--test-data-path", default=None, help="Override test CSV path.")
    parser.add_argument("--val-data-path", default=None, help="Override validation CSV path.")
    parser.add_argument("--output-dir", default=None, help="Output directory.")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level.")
    parser.add_argument("--bootstrap", type=int, default=5000, help="Bootstrap samples for paired mean-diff CI.")
    parser.add_argument("--bootstrap-seed", type=int, default=12345, help="Bootstrap RNG seed.")
    parser.add_argument("--probe-samples", type=int, default=100, help="Sample count for metric probe.")
    parser.add_argument("--skip-probe", action="store_true", help="Skip evaluator preflight.")
    parser.add_argument("--strict-metrics", action="store_true", help="Fail if any requested metric fails preflight.")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate existing sample CSVs.")
    parser.add_argument("--quiet", action="store_true", help="Suppress noisy generation/evaluator stdout.")
    parser.add_argument(
        "--missing-category-token",
        default=MISSING_CATEGORY_TOKEN,
        help=(
            "Token used only in evaluator inputs for categorical missing values. "
            "Use a token pandas will not parse as NA; default: __MISSING__."
        ),
    )
    parser.add_argument(
        "--no-normalize-missing-categories",
        action="store_false",
        dest="normalize_missing_categories",
        help="Disable categorical missing-value normalization before evaluation.",
    )
    parser.set_defaults(normalize_missing_categories=True)
    parser.add_argument(
        "--framework-evaluator",
        choices=["none", "csdeval", "plausibility", "both"],
        default=DEFAULT_FRAMEWORK_EVALUATOR,
        help="Optionally add parent-framework csdeval and/or plausibility scores to raw_metrics.csv.",
    )
    parser.add_argument(
        "--csdeval-info-path",
        default=None,
        help=(
            "Path to csdeval info.json. Defaults to ../../workdir/data/processed/<dataname>/info.json "
            "when present, otherwise a minimal info file is generated under the output directory."
        ),
    )
    parser.add_argument("--csdeval-config-path", default=None, help="Optional csdeval config YAML.")
    parser.add_argument(
        "--csdeval-skip-metric",
        action="append",
        default=[],
        help="Metric name to skip in csdeval. May be repeated. plausibility is always skipped inside csdeval.",
    )
    parser.add_argument(
        "--plausibility-model-dir",
        default=None,
        help="Plausibility model directory. If omitted, the parent framework dataset registry is consulted.",
    )
    parser.add_argument("--plausibility-batch-size", type=int, default=256)
    parser.add_argument(
        "--no-auto-process",
        action="store_true",
        help="Do not auto-run process_dataset.py when data/<dataname>/info.json is missing.",
    )
    return parser.parse_args()


def slugify(value: str) -> str:
    value = value.strip().replace(os.sep, "_")
    value = re.sub(r"[^A-Za-z0-9_.=-]+", "_", value)
    return value.strip("_") or "condition"


def parse_condition_item(raw: str) -> Condition:
    if "=" not in raw:
        raise ValueError(f"Condition must be column=value, got: {raw}")
    column, value = raw.split("=", 1)
    column = column.strip()
    value = value.strip()
    if not column or value == "":
        raise ValueError(f"Condition must have non-empty column and value, got: {raw}")
    return Condition(column=column, value=value)


def load_conditions(args: argparse.Namespace) -> list[Condition]:
    conditions = [parse_condition_item(item) for item in args.condition]
    if args.conditions_json:
        payload = json.loads(Path(args.conditions_json).read_text())
        if not isinstance(payload, list):
            raise ValueError("--conditions-json must contain a list")
        for item in payload:
            if not isinstance(item, dict) or "column" not in item or "value" not in item:
                raise ValueError("Each JSON condition must have column and value keys")
            conditions.append(Condition(column=str(item["column"]), value=str(item["value"])))
    if not conditions:
        raise ValueError("Provide at least one --condition or --conditions-json entry")
    return conditions


def default_output_dir(dataname: str) -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return Path("experiments") / f"conditional_significance_{dataname}_{stamp}"


def read_info(dataname: str) -> dict[str, Any]:
    info_path = Path("data") / dataname / "info.json"
    if not info_path.exists():
        template_path = Path("data") / "Info" / f"{dataname}.json"
        if template_path.exists():
            raise FileNotFoundError(
                f"Missing processed dataset info: {info_path}. "
                f"Template metadata exists at {template_path}; run "
                f"`python process_dataset.py --dataname {dataname}` first, or omit "
                "`--no-auto-process` so this script can do it."
            )
        raise FileNotFoundError(
            f"Missing dataset info: {info_path}; also checked data/Info/{dataname}.json"
        )
    info = json.loads(info_path.read_text())
    if not info.get("column_names"):
        raise ValueError(
            f"{info_path} has column_names=null. generate_conditional.py needs named columns."
        )
    return info


def processed_dataset_ready(dataname: str) -> bool:
    data_dir = Path("data") / dataname
    required = [
        data_dir / "info.json",
        data_dir / "X_num_train.npy",
        data_dir / "X_cat_train.npy",
        data_dir / "y_train.npy",
        data_dir / "train.csv",
    ]
    return all(path.exists() for path in required)


def validate_info_source_paths(dataname: str) -> None:
    template_path = Path("data") / "Info" / f"{dataname}.json"
    if not template_path.exists():
        raise FileNotFoundError(f"Cannot auto-process: missing {template_path}")
    info = json.loads(template_path.read_text())
    missing = []
    for key in ["data_path", "test_path", "val_path"]:
        raw_path = info.get(key)
        if raw_path and not Path(raw_path).exists():
            missing.append(f"{key}={raw_path}")
    if missing:
        raise FileNotFoundError(
            "Cannot auto-process because source CSV paths are missing: " + "; ".join(missing)
        )


def ensure_processed_dataset(args: argparse.Namespace) -> None:
    if processed_dataset_ready(args.dataname):
        return
    if args.no_auto_process:
        return

    validate_info_source_paths(args.dataname)
    data_dir = Path("data") / args.dataname
    data_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"Processed dataset data/{args.dataname}/info.json is missing; "
        f"running process_dataset.py --dataname {args.dataname}"
    )
    cmd = [sys.executable, "process_dataset.py", "--dataname", args.dataname]
    proc = subprocess.run(cmd, text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "Dataset auto-processing failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"stdout:\n{proc.stdout[-4000:]}\n"
            f"stderr:\n{proc.stderr[-4000:]}"
        )
    if not processed_dataset_ready(args.dataname):
        raise RuntimeError(
            f"Dataset auto-processing completed but required files are still missing under data/{args.dataname}"
        )


def resolve_data_paths(args: argparse.Namespace) -> tuple[Path, Path, Path | None]:
    real_path = Path(args.real_data_path) if args.real_data_path else first_existing_path(
        [Path("synthetic") / args.dataname / "real.csv", Path("data") / args.dataname / "train.csv"]
    )
    test_path = Path(args.test_data_path) if args.test_data_path else first_existing_path(
        [Path("synthetic") / args.dataname / "test.csv", Path("data") / args.dataname / "test.csv"]
    )
    val_path = Path(args.val_data_path) if args.val_data_path else first_existing_path(
        [Path("synthetic") / args.dataname / "val.csv", Path("data") / args.dataname / "val.csv"],
        required=False,
    )
    if not real_path.exists():
        raise FileNotFoundError(f"Missing real CSV for evaluation: {real_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Missing test CSV for evaluation: {test_path}")
    if not val_path.exists():
        val_path = None
    return real_path, test_path, val_path


def first_existing_path(paths: list[Path], required: bool = True) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0] if required else paths[0]


def validate_checkpoint_path(ckpt_path: str | None) -> None:
    if not ckpt_path:
        return
    path = Path(ckpt_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint does not exist: {ckpt_path}. Replace the placeholder with a real .pt file, "
            "for example one under tabdiff/ckpt/<dataname>/<exp_name>/."
        )


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested --device cuda, but torch.cuda.is_available() is false")
    return device


def resolve_num_samples(raw: str, real_path: Path) -> int:
    if raw == "same-as-real":
        return int(len(pd.read_csv(real_path)))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError("--num-samples must be an integer or 'same-as-real'") from exc
    if value <= 0:
        raise ValueError("--num-samples must be positive")
    return value


def validate_conditions(info: dict[str, Any], dataname: str, conditions: list[Condition]) -> None:
    columns = info["column_names"]
    missing = [cond.column for cond in conditions if cond.column not in columns]
    if missing:
        raise ValueError(f"Condition columns not found in info.json: {sorted(set(missing))}")

    train_csv = Path("data") / dataname / "train.csv"
    cat_encoders = info.get("cat_encoders") or {}
    cat_col_idx = set(info.get("cat_col_idx") or [])
    for cond in conditions:
        col_idx = columns.index(cond.column)
        if col_idx in cat_col_idx and cond.column not in cat_encoders and not train_csv.exists():
            raise FileNotFoundError(
                f"Condition column '{cond.column}' has no cat_encoders entry and {train_csv} is missing. "
                "generate_conditional.py falls back to that CSV for categorical value lookup."
            )


def categorical_column_names(info: dict[str, Any]) -> list[str]:
    columns = info["column_names"]
    cat_indices = list(info.get("cat_col_idx") or [])
    if info.get("task_type") != "regression":
        cat_indices.extend(info.get("target_col_idx") or [])
    names = []
    for idx in cat_indices:
        if 0 <= idx < len(columns):
            names.append(columns[idx])
    return names


def normalize_categorical_missing_values(
    df: pd.DataFrame,
    info: dict[str, Any],
    token: str,
    *,
    return_counts: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, int]]:
    """Replace categorical NA-like values with a stable non-NA token.

    TabDiff's Adult preprocessing stores missing categorical values as the
    literal string "nan". pandas parses that back as NA by default, while
    SDMetrics/rdt may later see synthetic-side "nan" as an unknown category.
    A non-NA sentinel keeps both sides in the same categorical space.
    """
    if not token or token.lower() in {"nan", "na", "null", "none"}:
        raise ValueError("--missing-category-token must not be parsed as a missing value")

    out = df.copy()
    counts: dict[str, int] = {}
    missing_like = {"", "?", "nan", "<na>", "null", "none"}
    for col in categorical_column_names(info):
        if col not in out.columns:
            continue
        series = out[col].astype("string")
        stripped = series.str.strip()
        mask = series.isna() | stripped.str.lower().isin(missing_like)
        count = int(mask.sum())
        if count:
            counts[col] = count
        out[col] = stripped.mask(mask, token).astype(str)

    if return_counts:
        return out, counts
    return out


def write_normalized_eval_csv(
    src_path: Path,
    dst_path: Path,
    info: dict[str, Any],
    token: str,
) -> dict[str, int]:
    df = pd.read_csv(src_path, keep_default_na=False)
    normalized, counts = normalize_categorical_missing_values(
        df,
        info,
        token,
        return_counts=True,
    )
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    normalized.to_csv(dst_path, index=False)
    return counts


def prepare_tabdiff_eval_inputs(
    args: argparse.Namespace,
    info: dict[str, Any],
    real_path: Path,
    test_path: Path,
    val_path: Path | None,
    out_dir: Path,
) -> tuple[Path, Path, Path | None, dict[str, Any]]:
    if not args.normalize_missing_categories:
        return real_path, test_path, val_path, {"enabled": False}

    eval_dir = out_dir / "eval_inputs"
    real_eval = eval_dir / "real.csv"
    test_eval = eval_dir / "test.csv"
    val_eval = eval_dir / "val.csv" if val_path else None
    counts = {
        "real": write_normalized_eval_csv(real_path, real_eval, info, args.missing_category_token),
        "test": write_normalized_eval_csv(test_path, test_eval, info, args.missing_category_token),
    }
    if val_path and val_eval:
        counts["val"] = write_normalized_eval_csv(val_path, val_eval, info, args.missing_category_token)

    report = {
        "enabled": True,
        "token": args.missing_category_token,
        "categorical_columns": categorical_column_names(info),
        "counts": counts,
        "paths": {
            "real": str(real_eval),
            "test": str(test_eval),
            "val": str(val_eval) if val_eval else None,
        },
    }
    (eval_dir / "missing_category_normalization.json").write_text(json.dumps(report, indent=2))
    return real_eval, test_eval, val_eval, report


def prepare_synthetic_for_eval(
    syn_df: pd.DataFrame,
    info: dict[str, Any],
    args: argparse.Namespace,
) -> pd.DataFrame:
    if not args.normalize_missing_categories:
        return syn_df.copy()
    return normalize_categorical_missing_values(
        syn_df,
        info,
        args.missing_category_token,
    )


def build_metrics(
    info: dict[str, Any],
    real_path: Path,
    test_path: Path,
    val_path: Path | None,
    device: str,
    metric_list: list[str] | None = None,
) -> TabMetrics:
    return TabMetrics(
        real_data_path=str(real_path),
        test_data_path=str(test_path),
        val_data_path=str(val_path) if val_path else None,
        info=info,
        device=device,
        metric_list=metric_list or [],
    )


def call_maybe_quiet(quiet: bool, func, *args, **kwargs):
    if not quiet:
        return func(*args, **kwargs)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return func(*args, **kwargs)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_samples(
    args: argparse.Namespace,
    condition: Condition,
    method: str,
    seed: int,
    num_samples: int,
    device: str,
    sample_path: Path,
) -> pd.DataFrame:
    if sample_path.exists() and not args.overwrite:
        return pd.read_csv(sample_path)

    set_seed(seed)
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    syn_df = call_maybe_quiet(
        args.quiet,
        generate_conditional_samples,
        dataname=args.dataname,
        condition_column=condition.column,
        condition_value=condition.value,
        num_samples=num_samples,
        w_num=args.w_num,
        w_cat=args.w_cat,
        resample_rounds=args.resample_rounds,
        ckpt_path=args.ckpt_path,
        device=device,
        privacy_method=method,
        stochastic_start_ratio=args.stochastic_start_ratio,
        stochastic_s_churn=args.stochastic_s_churn,
        stochastic_cat_noise_scale=args.stochastic_cat_noise_scale,
        midpoint_privacy_noise_scale=args.midpoint_privacy_noise_scale,
        midpoint_cat_noise_scale=args.midpoint_cat_noise_scale,
    )
    syn_df.to_csv(sample_path, index=False)
    return syn_df


def evaluate_one(
    evaluator: TabMetrics,
    syn_df: pd.DataFrame,
    metrics: list[str],
    quiet: bool,
) -> tuple[dict[str, float], dict[str, str]]:
    values: dict[str, float] = {}
    errors: dict[str, str] = {}
    for metric in metrics:
        func = getattr(evaluator, f"evaluate_{metric}")
        try:
            metric_values, _ = call_maybe_quiet(quiet, func, syn_df.copy())
        except Exception as exc:
            errors[metric] = f"{type(exc).__name__}: {exc}"
            continue
        for name, value in metric_values.items():
            if isinstance(value, (int, float, np.integer, np.floating)) and math.isfinite(float(value)):
                values[name] = float(value)
            else:
                errors[name] = f"non-finite/non-numeric value: {value!r}"
    return values, errors


def framework_enabled(args: argparse.Namespace, name: str) -> bool:
    return args.framework_evaluator in {name, "both"}


def ensure_framework_import_paths(out_dir: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-csdeval")
    log_dir = out_dir / "logs" / "plausibility"
    log_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("PLAUSIBILITY_LOG_DIR", str(log_dir))

    scripts_dir = (Path(__file__).resolve().parents[2] / "scripts").resolve()
    if scripts_dir.exists() and str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))


def minimal_csdeval_info(info: dict[str, Any]) -> dict[str, Any]:
    columns = info["column_names"]
    numerical_columns = [columns[i] for i in info.get("num_col_idx", [])]
    categorical_columns = [columns[i] for i in info.get("cat_col_idx", [])]
    target_column = None
    target_idx = info.get("target_col_idx") or []
    if target_idx:
        target_column = columns[target_idx[0]]
        if info.get("task_type") == "regression":
            numerical_columns.append(target_column)
        else:
            categorical_columns.append(target_column)
    payload = {
        "numerical_columns": numerical_columns,
        "categorical_columns": categorical_columns,
    }
    if target_column:
        payload["target_column"] = target_column
    return payload


def resolve_csdeval_info_path(args: argparse.Namespace, info: dict[str, Any], out_dir: Path) -> Path | None:
    if not framework_enabled(args, "csdeval"):
        return None
    if args.csdeval_info_path:
        path = Path(args.csdeval_info_path)
        if not path.exists():
            raise FileNotFoundError(f"--csdeval-info-path does not exist: {path}")
        return path

    framework_info = Path("../..") / "workdir" / "data" / "processed" / args.dataname / "info.json"
    if framework_info.exists():
        return framework_info

    generated = out_dir / "framework_inputs" / "csdeval_info.json"
    generated.parent.mkdir(parents=True, exist_ok=True)
    generated.write_text(json.dumps(minimal_csdeval_info(info), indent=2))
    return generated


def resolve_plausibility_model_dir(args: argparse.Namespace) -> Path | None:
    if not framework_enabled(args, "plausibility"):
        return None
    if args.plausibility_model_dir:
        path = Path(args.plausibility_model_dir)
        if not path.exists():
            raise FileNotFoundError(f"--plausibility-model-dir does not exist: {path}")
        return path

    try:
        import yaml
    except Exception:
        return None

    registry = Path("../..") / "config" / "datasets.yaml"
    if not registry.exists():
        return None
    payload = yaml.safe_load(registry.read_text()) or {}
    dataset_cfg = payload.get(args.dataname) or {}
    raw_path = dataset_cfg.get("plausibility_model")
    if not raw_path:
        return None
    candidate = (Path("../..") / raw_path).resolve()
    return candidate if candidate.exists() else None


def write_eval_sample_if_needed(
    syn_df: pd.DataFrame,
    eval_syn_df: pd.DataFrame,
    sample_path: Path,
    out_dir: Path,
    framework_is_enabled: bool,
) -> Path:
    if not framework_is_enabled:
        return sample_path
    rel = sample_path.relative_to(out_dir / "generated")
    eval_path = out_dir / "eval_generated" / rel
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    eval_syn_df.to_csv(eval_path, index=False)
    return eval_path


def extract_framework_scores(report: dict[str, Any], prefix: str) -> dict[str, float]:
    values: dict[str, float] = {}
    metrics = report.get("metrics", {}) if isinstance(report, dict) else {}
    if isinstance(metrics, dict):
        for name, payload in metrics.items():
            if not isinstance(payload, dict):
                continue
            score = payload.get("score")
            if isinstance(score, (int, float, np.integer, np.floating)) and math.isfinite(float(score)):
                values[f"{prefix}/{name}"] = float(score)

    composite = report.get("composite") if isinstance(report, dict) else None
    if isinstance(composite, dict):
        score = composite.get("score")
        if isinstance(score, (int, float, np.integer, np.floating)) and math.isfinite(float(score)):
            values[f"{prefix}/composite"] = float(score)
    return values


def evaluate_frameworks(
    args: argparse.Namespace,
    condition: Condition,
    real_eval_path: Path,
    train_eval_path: Path,
    eval_sample_path: Path,
    csdeval_info_path: Path | None,
    plausibility_model_dir: Path | None,
    report_path: Path,
) -> tuple[dict[str, float], dict[str, str]]:
    values: dict[str, float] = {}
    errors: dict[str, str] = {}
    report_payload: dict[str, Any] = {}

    if framework_enabled(args, "csdeval"):
        try:
            from evaluation_adapter import evaluate_condition_report

            skip_metrics = list(args.csdeval_skip_metric or [])
            report, error = call_maybe_quiet(
                args.quiet,
                evaluate_condition_report,
                real_path=str(real_eval_path),
                synthetic_path=str(eval_sample_path),
                condition=condition.label,
                info_path=str(csdeval_info_path),
                skip_metrics=skip_metrics,
                config_path=args.csdeval_config_path,
            )
            if error:
                errors["csdeval"] = error
            elif report is not None:
                report_payload["csdeval"] = report
                values.update(extract_framework_scores(report, "csdeval"))
        except Exception as exc:
            errors["csdeval"] = f"{type(exc).__name__}: {exc}"

    if framework_enabled(args, "plausibility"):
        if plausibility_model_dir is None:
            errors["plausibility"] = (
                "plausibility model not found; pass --plausibility-model-dir or add a valid "
                "plausibility_model path in ../../config/datasets.yaml"
            )
        else:
            try:
                from evaluation_adapter import compute_plausibility_score

                raw_nll, ratio, error = call_maybe_quiet(
                    args.quiet,
                    compute_plausibility_score,
                    input_path=str(eval_sample_path),
                    model_dir=str(plausibility_model_dir),
                    output_dir=str(report_path.parent / "plausibility_outputs"),
                    batch_size=args.plausibility_batch_size,
                    normalize_columns=True,
                    train_csv=str(train_eval_path),
                )
                report_payload["plausibility"] = {
                    "raw_nll": raw_nll,
                    "ratio": ratio,
                    "model_dir": str(plausibility_model_dir),
                }
                if error:
                    errors["plausibility"] = error
                if isinstance(raw_nll, (int, float, np.integer, np.floating)) and math.isfinite(float(raw_nll)):
                    values["plausibility/raw_nll"] = float(raw_nll)
                if isinstance(ratio, (int, float, np.integer, np.floating)) and math.isfinite(float(ratio)):
                    values["plausibility/ratio"] = float(ratio)
            except Exception as exc:
                errors["plausibility"] = f"{type(exc).__name__}: {exc}"

    if report_payload:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report_payload, indent=2, default=str))
    return values, errors


def probe_metrics(
    args: argparse.Namespace,
    conditions: list[Condition],
    requested_metrics: list[str],
    evaluator: TabMetrics,
    info: dict[str, Any],
    num_samples: int,
    device: str,
    out_dir: Path,
) -> tuple[list[str], dict[str, Any]]:
    if args.skip_probe:
        return requested_metrics, {"skipped": True, "enabled_metrics": requested_metrics, "disabled_metrics": {}}

    probe_condition = conditions[0]
    probe_samples = max(1, min(args.probe_samples, num_samples))
    probe_path = out_dir / "probe" / f"{probe_condition.slug}__{args.methods[0]}__seed{args.seed_start}.csv"
    syn_df = generate_samples(
        args,
        probe_condition,
        args.methods[0],
        args.seed_start,
        probe_samples,
        device,
        probe_path,
    )
    eval_syn_df = prepare_synthetic_for_eval(syn_df, info, args)
    values, errors = evaluate_one(evaluator, eval_syn_df, requested_metrics, args.quiet)
    enabled = []
    disabled: dict[str, str] = {}
    for metric in requested_metrics:
        metric_prefix = f"{metric}/"
        metric_worked = metric in values or any(name.startswith(metric_prefix) for name in values)
        if metric_worked:
            enabled.append(metric)
        else:
            disabled[metric] = errors.get(metric, "metric produced no numeric outputs")

    probe = {
        "skipped": False,
        "condition": probe_condition.label,
        "method": args.methods[0],
        "seed": args.seed_start,
        "sample_path": str(probe_path),
        "requested_metrics": requested_metrics,
        "enabled_metrics": enabled,
        "disabled_metrics": disabled,
        "probe_metric_values": values,
    }
    (out_dir / "metric_probe.json").write_text(json.dumps(probe, indent=2))

    if disabled and args.strict_metrics:
        raise RuntimeError(f"Metric preflight failed: {disabled}")
    if not enabled:
        raise RuntimeError(f"No requested metrics passed preflight: {disabled}")
    return enabled, probe


def run_experiment(
    args: argparse.Namespace,
    conditions: list[Condition],
    enabled_metrics: list[str],
    evaluator: TabMetrics,
    info: dict[str, Any],
    real_eval_path: Path,
    train_eval_path: Path,
    csdeval_info_path: Path | None,
    plausibility_model_dir: Path | None,
    num_samples: int,
    device: str,
    out_dir: Path,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    total = len(conditions) * args.repeats * len(args.methods)
    run_idx = 0

    for condition in conditions:
        for repeat in range(args.repeats):
            seed = args.seed_start + repeat
            for method in args.methods:
                run_idx += 1
                sample_path = (
                    out_dir
                    / "generated"
                    / args.dataname
                    / condition.slug
                    / f"seed_{seed}"
                    / f"{method}.csv"
                )
                print(
                    f"[{run_idx}/{total}] {args.dataname} {condition.label} "
                    f"seed={seed} method={method}"
                )
                try:
                    syn_df = generate_samples(args, condition, method, seed, num_samples, device, sample_path)
                except Exception as exc:
                    errors.append(
                        {
                            "stage": "generate",
                            "condition": condition.label,
                            "seed": seed,
                            "method": method,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    continue

                eval_syn_df = prepare_synthetic_for_eval(syn_df, info, args)
                eval_sample_path = write_eval_sample_if_needed(
                    syn_df,
                    eval_syn_df,
                    sample_path,
                    out_dir,
                    args.framework_evaluator != "none",
                )

                values, metric_errors = evaluate_one(evaluator, eval_syn_df, enabled_metrics, args.quiet)
                if metric_errors:
                    for metric, error in metric_errors.items():
                        errors.append(
                            {
                                "stage": "evaluate",
                                "condition": condition.label,
                                "seed": seed,
                                "method": method,
                                "metric": metric,
                                "error": error,
                            }
                        )

                framework_report_path = (
                    out_dir
                    / "framework_reports"
                    / args.dataname
                    / condition.slug
                    / f"seed_{seed}"
                    / f"{method}.json"
                )
                if args.framework_evaluator != "none":
                    framework_values, framework_errors = evaluate_frameworks(
                        args,
                        condition,
                        real_eval_path,
                        train_eval_path,
                        eval_sample_path,
                        csdeval_info_path,
                        plausibility_model_dir,
                        framework_report_path,
                    )
                    values.update(framework_values)
                    for metric, error in framework_errors.items():
                        errors.append(
                            {
                                "stage": "framework_evaluate",
                                "condition": condition.label,
                                "seed": seed,
                                "method": method,
                                "metric": metric,
                                "error": error,
                            }
                        )
                row = {
                    "dataname": args.dataname,
                    "condition_column": condition.column,
                    "condition_value": condition.value,
                    "condition_id": condition.slug,
                    "seed": seed,
                    "repeat": repeat,
                    "method": method,
                    "num_samples": len(syn_df),
                    "sample_path": str(sample_path),
                    "eval_sample_path": str(eval_sample_path),
                    "framework_report_path": str(framework_report_path)
                    if framework_report_path.exists()
                    else "",
                }
                row.update(values)
                rows.append(row)

                pd.DataFrame(rows).to_csv(out_dir / "raw_metrics.csv", index=False)
                if errors:
                    pd.DataFrame(errors).to_csv(out_dir / "run_errors.csv", index=False)

    return pd.DataFrame(rows), errors


def numeric_metric_columns(df: pd.DataFrame) -> list[str]:
    return [
        col
        for col in df.columns
        if col not in ID_COLUMNS and pd.api.types.is_numeric_dtype(df[col])
    ]


def bootstrap_mean_ci(diff: np.ndarray, n_boot: int, alpha: float, seed: int) -> tuple[float, float]:
    diff = np.asarray(diff, dtype=float)
    diff = diff[np.isfinite(diff)]
    if diff.size == 0:
        return (math.nan, math.nan)
    if diff.size == 1 or n_boot <= 0:
        return (float(diff.mean()), float(diff.mean()))
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample = rng.choice(diff, size=diff.size, replace=True)
        means[i] = sample.mean()
    low, high = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(low), float(high)


def holm_adjust(p_values: list[float]) -> list[float]:
    adjusted = [math.nan] * len(p_values)
    valid = [(i, p) for i, p in enumerate(p_values) if p is not None and math.isfinite(p)]
    valid.sort(key=lambda item: item[1])
    running_max = 0.0
    m = len(valid)
    for rank, (idx, p) in enumerate(valid):
        adj = min(1.0, (m - rank) * p)
        running_max = max(running_max, adj)
        adjusted[idx] = running_max
    return adjusted


def compute_global_tests(
    df: pd.DataFrame,
    methods: list[str],
    metric_cols: list[str],
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    if stats is None:
        return pd.DataFrame()
    group_cols = group_cols or []
    rows = []
    grouped = [((), df)] if not group_cols else df.groupby(group_cols, dropna=False)
    for group_key, group_df in grouped:
        group_values = {}
        if group_cols:
            if not isinstance(group_key, tuple):
                group_key = (group_key,)
            group_values = dict(zip(group_cols, group_key))
        for metric in metric_cols:
            pivot = group_df.pivot_table(
                index=["condition_id", "seed"],
                columns="method",
                values=metric,
                aggfunc="mean",
            )
            available_methods = [method for method in methods if method in pivot.columns]
            pivot = pivot.dropna(subset=available_methods)
            if len(available_methods) < 3 or len(pivot) < 2:
                rows.append({**group_values, "metric": metric, "n_blocks": len(pivot), "test": "friedman", "p_value": math.nan, "statistic": math.nan, "error": "not enough complete paired blocks"})
                continue
            arrays = [pivot[m].to_numpy(dtype=float) for m in available_methods]
            if all(np.allclose(arrays[0], arr) for arr in arrays[1:]):
                rows.append(
                    {
                        **group_values,
                        "metric": metric,
                        "methods": ",".join(available_methods),
                        "n_blocks": len(pivot),
                        "test": "friedman",
                        "statistic": 0.0,
                        "p_value": 1.0,
                        "error": "",
                    }
                )
                continue
            try:
                statistic, p_value = stats.friedmanchisquare(*arrays)
                error = ""
            except Exception as exc:
                statistic, p_value, error = math.nan, math.nan, f"{type(exc).__name__}: {exc}"
            rows.append(
                {
                    **group_values,
                    "metric": metric,
                    "methods": ",".join(available_methods),
                    "n_blocks": len(pivot),
                    "test": "friedman",
                    "statistic": statistic,
                    "p_value": p_value,
                    "error": error,
                }
            )
    return pd.DataFrame(rows)


def compute_pairwise_tests(
    df: pd.DataFrame,
    methods: list[str],
    metric_cols: list[str],
    alpha: float,
    n_boot: int,
    boot_seed: int,
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    group_cols = group_cols or []
    rows = []
    grouped = [((), df)] if not group_cols else df.groupby(group_cols, dropna=False)
    for group_key, group_df in grouped:
        group_values = {}
        if group_cols:
            if not isinstance(group_key, tuple):
                group_key = (group_key,)
            group_values = dict(zip(group_cols, group_key))

        metric_row_indices: dict[str, list[int]] = {metric: [] for metric in metric_cols}
        for metric in metric_cols:
            pivot = group_df.pivot_table(
                index=["condition_id", "seed"],
                columns="method",
                values=metric,
                aggfunc="mean",
            )
            for method_a, method_b in combinations(methods, 2):
                if method_a not in pivot.columns or method_b not in pivot.columns:
                    continue
                paired = pivot[[method_a, method_b]].dropna()
                diff = (paired[method_b] - paired[method_a]).to_numpy(dtype=float)
                diff = diff[np.isfinite(diff)]
                mean_a = float(paired[method_a].mean()) if len(paired) else math.nan
                mean_b = float(paired[method_b].mean()) if len(paired) else math.nan
                mean_diff = float(diff.mean()) if diff.size else math.nan
                std_diff = float(diff.std(ddof=1)) if diff.size > 1 else math.nan
                cohen_dz = (
                    mean_diff / std_diff
                    if std_diff and math.isfinite(std_diff) and abs(std_diff) > 1e-12
                    else math.nan
                )
                ci_low, ci_high = bootstrap_mean_ci(
                    diff,
                    n_boot=n_boot,
                    alpha=alpha,
                    seed=stable_seed(boot_seed, metric, method_a, method_b, str(group_values)),
                )

                if stats is None:
                    statistic, p_value, error = math.nan, math.nan, "scipy unavailable"
                elif diff.size < 2:
                    statistic, p_value, error = math.nan, math.nan, "not enough complete paired blocks"
                elif np.allclose(diff, 0):
                    statistic, p_value, error = 0.0, 1.0, ""
                else:
                    try:
                        statistic, p_value = stats.wilcoxon(diff, zero_method="wilcox", alternative="two-sided")
                        error = ""
                    except Exception as exc:
                        statistic, p_value, error = math.nan, math.nan, f"{type(exc).__name__}: {exc}"

                row_idx = len(rows)
                metric_row_indices[metric].append(row_idx)
                rows.append(
                    {
                        **group_values,
                        "metric": metric,
                        "method_a": method_a,
                        "method_b": method_b,
                        "comparison": f"{method_b} - {method_a}",
                        "n_blocks": int(diff.size),
                        "mean_a": mean_a,
                        "mean_b": mean_b,
                        "mean_diff": mean_diff,
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                        "cohen_dz": cohen_dz,
                        "test": "wilcoxon_signed_rank",
                        "statistic": float(statistic) if statistic is not None and math.isfinite(float(statistic)) else math.nan,
                        "p_value": float(p_value) if p_value is not None and math.isfinite(float(p_value)) else math.nan,
                        "error": error,
                    }
                )

        for metric, indices in metric_row_indices.items():
            adjusted = holm_adjust([rows[i]["p_value"] for i in indices])
            for idx, p_holm in zip(indices, adjusted):
                rows[idx]["p_holm"] = p_holm
                rows[idx]["alpha"] = alpha
                rows[idx]["significant"] = bool(
                    math.isfinite(p_holm)
                    and p_holm < alpha
                    and math.isfinite(rows[idx]["ci_low"])
                    and math.isfinite(rows[idx]["ci_high"])
                    and not (rows[idx]["ci_low"] <= 0 <= rows[idx]["ci_high"])
                )
    return pd.DataFrame(rows)


def stable_seed(base: int, *parts: str) -> int:
    digest = hashlib.sha256("||".join(parts).encode("utf-8")).hexdigest()
    return (base + int(digest[:8], 16)) % (2**32)


def summarize(df: pd.DataFrame, metric_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    agg = ["count", "mean", "std", "median", "min", "max"]
    by_method = df.groupby("method", dropna=False)[metric_cols].agg(agg)
    by_method.columns = ["__".join(col).strip("_") for col in by_method.columns.to_flat_index()]
    by_method = by_method.reset_index()

    by_condition = df.groupby(["condition_id", "condition_column", "condition_value", "method"], dropna=False)[metric_cols].agg(agg)
    by_condition.columns = ["__".join(col).strip("_") for col in by_condition.columns.to_flat_index()]
    by_condition = by_condition.reset_index()
    return by_method, by_condition


def write_report(
    out_dir: Path,
    args: argparse.Namespace,
    enabled_metrics: list[str],
    probe: dict[str, Any],
    metric_cols: list[str],
    summary_by_method: pd.DataFrame,
    global_tests: pd.DataFrame,
    pairwise_tests: pd.DataFrame,
    errors: list[dict[str, Any]],
) -> None:
    report_path = out_dir / "report.md"
    lines = [
        "# Conditional Method Significance Report",
        "",
        f"- Dataset: `{args.dataname}`",
        f"- Methods: `{', '.join(args.methods)}`",
        f"- Enabled evaluators: `{', '.join(enabled_metrics)}`",
        f"- Framework evaluator: `{args.framework_evaluator}`",
        f"- Categorical missing normalization: `{args.normalize_missing_categories}`",
        f"- Numeric metric columns: `{', '.join(metric_cols)}`",
        f"- Repeats: `{args.repeats}`",
        f"- Alpha: `{args.alpha}`",
        f"- Bootstrap samples: `{args.bootstrap}`",
        "",
        "## Metric Probe",
        "",
        f"- Probe skipped: `{probe.get('skipped', False)}`",
        f"- Disabled metrics: `{json.dumps(probe.get('disabled_metrics', {}), sort_keys=True)}`",
        "",
        "## Summary By Method",
        "",
    ]
    markdown_table(summary_by_method, lines, max_rows=20)
    lines.extend(["", "## Global Tests", ""])
    markdown_table(global_tests, lines, max_rows=50)
    lines.extend(["", "## Significant Pairwise Differences", ""])
    if pairwise_tests.empty:
        lines.append("No pairwise tests were produced.")
    else:
        sig = pairwise_tests[pairwise_tests.get("significant", False) == True]  # noqa: E712
        if sig.empty:
            lines.append("No pairwise metric differences met the configured significance rule.")
        else:
            cols = [
                col
                for col in [
                    "metric",
                    "comparison",
                    "n_blocks",
                    "mean_diff",
                    "ci_low",
                    "ci_high",
                    "p_value",
                    "p_holm",
                    "cohen_dz",
                ]
                if col in sig.columns
            ]
            markdown_table(sig[cols], lines, max_rows=100)
    lines.extend(
        [
            "",
            "## Run Errors",
            "",
            f"- Error count: `{len(errors)}`",
            "",
            "See `raw_metrics.csv`, `pairwise_tests.csv`, `global_tests.csv`, "
            "`summary_by_method.csv`, and `run_errors.csv` for machine-readable details.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines))


def markdown_table(df: pd.DataFrame, lines: list[str], max_rows: int) -> None:
    if df is None or df.empty:
        lines.append("_No rows._")
        return
    view = df.head(max_rows).copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{x:.6g}")
    lines.append(view.to_markdown(index=False))
    if len(df) > max_rows:
        lines.append(f"\n_Showing {max_rows} of {len(df)} rows._")


def main() -> int:
    args = parse_args()
    if args.repeats <= 0:
        raise ValueError("--repeats must be positive")
    if not (0 < args.alpha < 1):
        raise ValueError("--alpha must be in (0, 1)")

    validate_checkpoint_path(args.ckpt_path)
    ensure_processed_dataset(args)
    conditions = load_conditions(args)
    info = read_info(args.dataname)
    validate_conditions(info, args.dataname, conditions)
    real_path, test_path, val_path = resolve_data_paths(args)
    num_samples = resolve_num_samples(args.num_samples, real_path)
    device = resolve_device(args.device)
    out_dir = Path(args.output_dir) if args.output_dir else default_output_dir(args.dataname)
    out_dir.mkdir(parents=True, exist_ok=True)
    real_eval_path, test_eval_path, val_eval_path, missing_norm_report = prepare_tabdiff_eval_inputs(
        args,
        info,
        real_path,
        test_path,
        val_path,
        out_dir,
    )
    if args.framework_evaluator != "none":
        ensure_framework_import_paths(out_dir)
    csdeval_info_path = resolve_csdeval_info_path(args, info, out_dir)
    plausibility_model_dir = resolve_plausibility_model_dir(args)

    config = {
        **vars(args),
        "conditions": [condition.__dict__ for condition in conditions],
        "real_data_path": str(real_path),
        "test_data_path": str(test_path),
        "val_data_path": str(val_path) if val_path else None,
        "eval_real_data_path": str(real_eval_path),
        "eval_test_data_path": str(test_eval_path),
        "eval_val_data_path": str(val_eval_path) if val_eval_path else None,
        "missing_category_normalization": missing_norm_report,
        "csdeval_info_path": str(csdeval_info_path) if csdeval_info_path else None,
        "plausibility_model_dir": str(plausibility_model_dir) if plausibility_model_dir else None,
        "resolved_num_samples": num_samples,
        "resolved_device": device,
        "output_dir": str(out_dir),
        "scipy_available": stats is not None,
    }
    (out_dir / "run_config.json").write_text(json.dumps(config, indent=2))

    evaluator = build_metrics(info, real_eval_path, test_eval_path, val_eval_path, device)
    enabled_metrics, probe = probe_metrics(
        args,
        conditions,
        args.metrics,
        evaluator,
        info,
        num_samples,
        device,
        out_dir,
    )
    if args.skip_probe:
        (out_dir / "metric_probe.json").write_text(json.dumps(probe, indent=2))

    raw_df, errors = run_experiment(
        args,
        conditions,
        enabled_metrics,
        evaluator,
        info,
        real_eval_path,
        real_eval_path,
        csdeval_info_path,
        plausibility_model_dir,
        num_samples,
        device,
        out_dir,
    )
    if raw_df.empty:
        raise RuntimeError("No successful runs produced metrics")

    metric_cols = numeric_metric_columns(raw_df)
    if not metric_cols:
        raise RuntimeError("No numeric metric columns were produced")

    summary_by_method, summary_by_condition = summarize(raw_df, metric_cols)
    global_tests = compute_global_tests(raw_df, args.methods, metric_cols)
    global_tests_by_condition = compute_global_tests(
        raw_df,
        args.methods,
        metric_cols,
        group_cols=["condition_id", "condition_column", "condition_value"],
    )
    pairwise_tests = compute_pairwise_tests(
        raw_df,
        args.methods,
        metric_cols,
        args.alpha,
        args.bootstrap,
        args.bootstrap_seed,
    )
    pairwise_tests_by_condition = compute_pairwise_tests(
        raw_df,
        args.methods,
        metric_cols,
        args.alpha,
        args.bootstrap,
        args.bootstrap_seed,
        group_cols=["condition_id", "condition_column", "condition_value"],
    )

    raw_df.to_csv(out_dir / "raw_metrics.csv", index=False)
    summary_by_method.to_csv(out_dir / "summary_by_method.csv", index=False)
    summary_by_condition.to_csv(out_dir / "summary_by_condition_method.csv", index=False)
    global_tests.to_csv(out_dir / "global_tests.csv", index=False)
    global_tests_by_condition.to_csv(out_dir / "global_tests_by_condition.csv", index=False)
    pairwise_tests.to_csv(out_dir / "pairwise_tests.csv", index=False)
    pairwise_tests_by_condition.to_csv(out_dir / "pairwise_tests_by_condition.csv", index=False)
    pd.DataFrame(
        errors,
        columns=["stage", "condition", "seed", "method", "metric", "error"],
    ).to_csv(out_dir / "run_errors.csv", index=False)

    write_report(
        out_dir,
        args,
        enabled_metrics,
        probe,
        metric_cols,
        summary_by_method,
        global_tests,
        pairwise_tests,
        errors,
    )

    print(f"\nWrote experiment outputs to: {out_dir}")
    print(f"Enabled evaluators: {', '.join(enabled_metrics)}")
    print("Main files: raw_metrics.csv, summary_by_method.csv, global_tests.csv, pairwise_tests.csv, report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
