"""TabDiff engine — ported from the host's wrappers/tabdiff_wrapper.py.

Standalone (no host `wrappers/` or `scripts/` imports) so the plugin is
self-contained inside the submodule. Training is unconditional (TabDiff's own
CLI); equality constraints are applied at inference via REPAINT-style
imputation (`sample_impute`), so train/generate decouple: train once ->
model_dir, then generate under N constraints from the same dir without
retraining.

The model_dir holds two subdirs that are symlinked into TabDiff's expected
layout for both train and generate:
    model_dir/data  <- TABDIFF_DIR/data/<dataset>_hashed        (info.json etc.)
    model_dir/ckpt  <- TABDIFF_DIR/tabdiff/ckpt/<dataset>_hashed (.pt files)
"""
from __future__ import annotations

import datetime
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

TABDIFF_DIR = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, TABDIFF_DIR)

logger = logging.getLogger("tabdiff.sdcontract")

META_NAME = "meta.json"
TRAIN_LOCK_DIR = ".train_lock"

# Privacy-mode profiles (mirror wrappers/tabdiff_wrapper.py main()).
_S_CHURN_DEFAULT = 40.0
_PRIVACY_NOISE_DEFAULT = 0.2


# --- inlined link helpers (were wrappers.utils.common) ----------------------

def _ensure_dir_link(src: str, dst: str) -> None:
    """Symlink dst -> src (a directory), replacing whatever is at dst."""
    if os.path.islink(dst) or os.path.isfile(dst):
        os.unlink(dst)
    elif os.path.isdir(dst):
        import shutil
        shutil.rmtree(dst)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    os.symlink(os.path.abspath(src), dst)


def _ensure_file_link(src: str, dst: str) -> None:
    """Symlink dst -> src (a file), replacing whatever is at dst."""
    if os.path.islink(dst) or os.path.exists(dst):
        os.unlink(dst)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    os.symlink(os.path.abspath(src), dst)


class _ModelDirLock:
    def __init__(self, model_dir, poll_seconds: float = 1.0, timeout_seconds: float = 1800.0):
        self.lock_dir = Path(model_dir) / TRAIN_LOCK_DIR
        self.poll_seconds = poll_seconds
        self.timeout_seconds = timeout_seconds

    def __enter__(self):
        deadline = time.monotonic() + self.timeout_seconds
        while True:
            try:
                os.mkdir(self.lock_dir)
                return self
            except FileExistsError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"Could not acquire training lock {self.lock_dir} after "
                        f"{self.timeout_seconds:.0f}s. If no other process is running, "
                        f"delete the lock directory manually and retry."
                    )
                time.sleep(self.poll_seconds)

    def __exit__(self, exc_type, exc, tb):
        try:
            os.rmdir(self.lock_dir)
        except FileNotFoundError:
            pass


def _resolve_device(device: str | None) -> str:
    if device and device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA not available. Falling back to CPU.")
        return "cpu"
    if not device:
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


# --- decode / sampling (ported verbatim from tabdiff_wrapper.py) ------------

def custom_decode_synthetic_data(syn_data, info, num_inverse, int_inverse, cat_inverse):
    """Decode synthetic data back to original feature space using stored inverses."""
    from tabdiff.trainer import split_num_cat_target, recover_data

    syn_num, syn_cat, syn_target = split_num_cat_target(
        syn_data, info, num_inverse, int_inverse, cat_inverse
    )
    syn_df = recover_data(syn_num, syn_cat, syn_target, info)

    idx_name_mapping = {int(k): v for k, v in info["idx_name_mapping"].items()}
    syn_df.rename(columns=idx_name_mapping, inplace=True)

    int_col_names = [info["column_names"][i] for i in info.get("int_col_idx", [])]
    for col_name in int_col_names:
        if col_name in syn_df.columns:
            numeric_series = pd.to_numeric(syn_df[col_name], errors="coerce").round()
            if numeric_series.isnull().any():
                syn_df[col_name] = numeric_series
            else:
                syn_df[col_name] = numeric_series.astype(int)

    cat_indices = list(info.get("cat_col_idx", []))
    if info["task_type"] != "regression":
        cat_indices += info.get("target_col_idx", [])

    if "cat_encoders" not in info:
        raise RuntimeError(
            "info.json is missing 'cat_encoders'. The data directory contains a "
            "stale or externally-generated info.json. Delete the model directory "
            "and re-run to trigger re-training with --data-csv, which will write "
            "a correct info.json."
        )

    for idx in cat_indices:
        col_name = info["column_names"][idx]
        if col_name not in syn_df.columns:
            continue
        syn_df[col_name] = syn_df[col_name].apply(lambda x: x.strip() if isinstance(x, str) else x)

    return syn_df


def generate_multi_conditional(
    dataname="adult",
    constraints=None,  # list of (col, val) tuples
    num_samples=100,
    w_num=0.6,
    w_cat=0.6,
    resample_rounds=1,
    ckpt_path=None,
    device="cuda",
    stochastic_start_ratio=1.0,
    s_churn=0,
    privacy_noise_scale=0.0,
    cat_noise_scale=0.0,
    impute_condition="x_t",
    num_inference_steps=None,
):
    original_cwd = os.getcwd()
    os.chdir(TABDIFF_DIR)
    try:
        from generate_conditional import load_model_and_info, get_category_encoding

        if constraints is None:
            constraints = []

        logger.info("Loading model for %s...", dataname)
        device = _resolve_device(device)

        (
            diffusion, info, X_num_train, X_cat_train, d_numerical, categories,
            num_inverse, int_inverse, cat_inverse, num_transform, int_transform,
        ) = load_model_and_info(dataname, ckpt_path=ckpt_path, device=device)

        if num_inference_steps is not None:
            logger.info("Overriding num_timesteps from %d to %d", diffusion.num_timesteps, num_inference_steps)
            diffusion.num_timesteps = num_inference_steps

        x_num = torch.randn(num_samples, d_numerical).to(device) * 0.5
        x_cat = torch.zeros(num_samples, len(categories)).long().to(device)
        num_mask_idx = list(range(d_numerical))
        cat_mask_idx = list(range(len(categories)))

        logger.info("Applying %d constraints:", len(constraints))
        for col_name, col_val in constraints:
            original_col_name = col_name
            if col_name not in info["column_names"]:
                for alt in [col_name.replace("-", "."), col_name.replace(" ", "."),
                            col_name.replace("-", "_"), col_name.replace(" ", "_")]:
                    if alt in info["column_names"]:
                        col_name = alt
                        break
            if col_name not in info["column_names"]:
                raise ValueError(
                    f"Column '{original_col_name}' not found in TabDiff info['column_names']."
                )

            col_idx = info["column_names"].index(col_name)
            is_numerical = col_idx in info["num_col_idx"]
            is_categorical = col_idx in info["cat_col_idx"]
            if not (is_numerical or is_categorical):
                if col_idx in info.get("target_col_idx", []):
                    is_categorical = True
                else:
                    raise ValueError(f"Column {col_name} is not numerical, categorical, or target.")

            if is_categorical:
                cat_position = -1
                if col_idx in info["cat_col_idx"]:
                    cat_position = info["cat_col_idx"].index(col_idx)
                    if info["task_type"] != "regression":
                        cat_position += 1
                elif col_idx in info.get("target_col_idx", []):
                    cat_position = 0
                if cat_position == -1:
                    raise ValueError(f"Could not resolve categorical position for column '{col_name}'.")

                if col_val == "?" and col_val not in info.get("cat_encoders", {}).get(col_name, []):
                    if "nan" in info.get("cat_encoders", {}).get(col_name, []):
                        logger.debug("Mapping '%s=?' to 'nan' for TabDiff compatibility.", original_col_name)
                        col_val = "nan"

                encoded_value, _, _ = get_category_encoding(dataname, col_name, col_val)
                x_cat[:, cat_position] = encoded_value
                if cat_position in cat_mask_idx:
                    cat_mask_idx.remove(cat_position)
                logger.debug("  - %s = %s (Encoded: %s, CatIdx: %d)", col_name, col_val, encoded_value, cat_position)
            else:
                num_position = info["num_col_idx"].index(col_idx)
                try:
                    numeric_value = float(col_val)
                except (ValueError, TypeError):
                    raise ValueError(f"Value '{col_val}' for numerical column '{col_name}' must be a number.")

                base_row = X_num_train.mean(axis=0).astype(np.float32)
                row = base_row.copy()
                row[num_position] = numeric_value
                row = row.reshape(1, -1)
                if int_transform is not None:
                    row = int_transform.transform(row)
                if num_transform is not None:
                    row = num_transform.transform(row)
                normalized_value = row[0, num_position]
                x_num[:, num_position] = torch.tensor(normalized_value, device=device, dtype=torch.float32)
                if num_position in num_mask_idx:
                    num_mask_idx.remove(num_position)
                logger.debug("  - %s = %s (Normalized: %.4f, NumIdx: %d)", col_name, col_val, normalized_value, num_position)

        logger.info("Generation started...")
        with torch.no_grad():
            syn_data = diffusion.sample_impute(
                x_num, x_cat, num_mask_idx, cat_mask_idx,
                resample_rounds=resample_rounds, impute_condition=impute_condition,
                w_num=w_num, w_cat=w_cat, stochastic_start_ratio=stochastic_start_ratio,
                s_churn=s_churn, privacy_noise_scale=privacy_noise_scale, cat_noise_scale=cat_noise_scale,
            )
        return custom_decode_synthetic_data(syn_data, info, num_inverse, int_inverse, cat_inverse)
    finally:
        os.chdir(original_cwd)


def generate_unconditional(dataname="adult", num_samples=100, ckpt_path=None,
                           device="cuda", num_inference_steps=None):
    """Vanilla unconditional TabDiff sampling via diffusion.sample_all()."""
    original_cwd = os.getcwd()
    os.chdir(TABDIFF_DIR)
    try:
        from generate_conditional import load_model_and_info

        device = _resolve_device(device)
        (
            diffusion, info, _Xn, _Xc, _dn, _cats,
            num_inverse, int_inverse, cat_inverse, _nt, _it,
        ) = load_model_and_info(dataname, ckpt_path=ckpt_path, device=device)

        if num_inference_steps is not None:
            logger.info("Overriding num_timesteps from %d to %d", diffusion.num_timesteps, num_inference_steps)
            diffusion.num_timesteps = num_inference_steps

        logger.info("Unconditional generation started (sample_all)...")
        with torch.no_grad():
            syn_data = diffusion.sample_all(num_samples, batch_size=num_samples)
        return custom_decode_synthetic_data(syn_data, info, num_inverse, int_inverse, cat_inverse)
    finally:
        os.chdir(original_cwd)


def generate_guided(
    dataname="adult",
    constraint_specs=None,   # list of native dialect strings, e.g. ["age>30", "education=Bachelors"]
    num_samples=100,
    ckpt_path=None,
    device="cuda",
    num_inference_steps=None,
    num_scale=0.5, cat_scale=4.0, mean_scale=0.1,
    backward_steps=10, backward_lr=1.0, guidance_schedule="none",
    cat_snap_final=False,
):
    """tabdiff-universal: training-free constraint guidance over TabDiff's model.
    Numeric constraints guide the x0 estimate; categorical constraints bias the
    unmasking logits. Falls back to nothing-to-guide → unconditional handled by caller."""
    from tabdiff.guidance import (
        NumericConstraint, CategoricalConstraint, parse_constraint_spec,
    )

    original_cwd = os.getcwd()
    os.chdir(TABDIFF_DIR)
    try:
        from generate_conditional import load_model_and_info

        device = _resolve_device(device)
        (
            diffusion, info, X_num_train, _Xc, _dn, _cats,
            num_inverse, int_inverse, cat_inverse, num_transform, int_transform,
        ) = load_model_and_info(dataname, ckpt_path=ckpt_path, device=device)

        if num_inference_steps is not None:
            diffusion.num_timesteps = num_inference_steps

        # Class lists for categorical/target columns missing from info["cat_encoders"]
        # (the target column is omitted there) — fall back to train.csv unique values,
        # mirroring generate_conditional.get_category_encoding.
        cat_classes = {}
        encoders = info.get("cat_encoders", {})
        train_df = None
        for ci in list(info.get("cat_col_idx", [])) + list(info.get("target_col_idx", [])):
            cname = info["column_names"][ci]
            if cname not in encoders:
                if train_df is None:
                    train_df = pd.read_csv(f"data/{dataname}/train.csv")
                cat_classes[cname] = sorted(train_df[cname].unique().tolist())

        parsed = [
            parse_constraint_spec(
                s, info, X_num_train, num_transform, int_transform,
                num_scale=num_scale, cat_scale=cat_scale, mean_scale=mean_scale,
                cat_classes=cat_classes,
            )
            for s in (constraint_specs or [])
        ]
        num_constraints = [c for c in parsed if isinstance(c, NumericConstraint)]
        cat_constraints = [c for c in parsed if isinstance(c, CategoricalConstraint)]
        logger.info("Guided generation: %d numeric, %d categorical constraints",
                    len(num_constraints), len(cat_constraints))

        with torch.no_grad():
            syn_data = diffusion.sample_guided(
                num_samples,
                num_constraints=num_constraints, cat_constraints=cat_constraints,
                backward_steps=backward_steps, backward_lr=backward_lr,
                guidance_schedule=guidance_schedule, cat_snap_final=cat_snap_final,
            )
        return custom_decode_synthetic_data(syn_data, info, num_inverse, int_inverse, cat_inverse)
    finally:
        os.chdir(original_cwd)


# --- checkpoint discovery / training (ported) -------------------------------

def _tabdiff_checkpoint_dirs(dataname):
    ckpt_root = os.path.join(TABDIFF_DIR, "tabdiff", "ckpt", dataname)
    dirs = []
    default_dir = os.path.join(ckpt_root, "learnable_schedule")
    if os.path.isdir(default_dir):
        dirs.append(default_dir)
    if os.path.isdir(ckpt_root):
        for entry in sorted(os.listdir(ckpt_root)):
            path = os.path.join(ckpt_root, entry)
            if os.path.isdir(path) and path not in dirs:
                dirs.append(path)
    return dirs


def _tabdiff_find_checkpoint(dataname):
    for ckpt_dir in _tabdiff_checkpoint_dirs(dataname):
        ckpts = [f for f in os.listdir(ckpt_dir) if f.endswith(".pt") and "model" in f.lower()]
        if not ckpts:
            continue
        best = ([f for f in ckpts if f.startswith("best_ema_model")]
                or [f for f in ckpts if f.startswith("best_model")] or ckpts)
        return os.path.join(ckpt_dir, sorted(best)[-1])
    return None


def _tabdiff_has_checkpoint(dataname):
    return _tabdiff_find_checkpoint(dataname) is not None


def _train_tabdiff_model(dataname, train_csv=None, label_col=None, gpu_index=0,
                         num_epochs=None, check_val_every=None):
    base_cmd = [
        sys.executable, os.path.join(TABDIFF_DIR, "main.py"),
        "--dataname", dataname, "--mode", "train", "--method", "tabdiff",
        "--gpu", str(gpu_index), "--no_wandb",
    ]
    if num_epochs is not None:
        base_cmd.extend(["--steps", str(num_epochs)])
    if check_val_every is not None:
        base_cmd.extend(["--check_val_every", str(check_val_every)])
    if train_csv:
        base_cmd.extend(["--data-csv", os.path.abspath(train_csv)])

    label_arg_variants = ["--label-col", "--label_col"] if label_col else [None]
    last_result = None
    for label_arg in label_arg_variants:
        cmd = list(base_cmd)
        if label_arg:
            cmd.extend([label_arg, label_col])
        logger.info("Training TabDiff model for %s via CLI...", dataname)
        result = subprocess.run(cmd, cwd=TABDIFF_DIR, text=True, capture_output=True)
        last_result = result
        if result.stdout:
            logger.info(result.stdout.rstrip())
        if result.stderr:
            logger.warning(result.stderr.rstrip())

        ckpt_path = _tabdiff_find_checkpoint(dataname)
        if ckpt_path:
            if result.returncode != 0:
                logger.warning(
                    "TabDiff training exited with code %d, but a checkpoint exists at %s. Continuing.",
                    result.returncode, ckpt_path,
                )
            return
        parse_error = "unrecognized arguments" in f"{result.stdout}\n{result.stderr}"
        if not (label_arg == "--label-col" and parse_error):
            break

    searched = _tabdiff_checkpoint_dirs(dataname)
    raise RuntimeError(
        "TabDiff training did not produce a checkpoint in "
        f"{searched or [os.path.join(TABDIFF_DIR, 'tabdiff', 'ckpt', dataname)]} "
        f"(exit code {last_result.returncode if last_result else 'unknown'})."
    )


# --- link layout shared by train + generate ---------------------------------

def _link_model_dir(abs_model_dir: str, dataset: str) -> str:
    """Create model_dir/{data,ckpt} and symlink them into TabDiff's layout.

    Returns the hashed dataname TabDiff sees (``<dataset>_hashed``).
    """
    hashed = f"{dataset}_hashed"

    data_source_dir = os.path.join(abs_model_dir, "data")
    if not os.path.isdir(data_source_dir):
        if os.path.isfile(os.path.join(abs_model_dir, "info.json")):
            data_source_dir = abs_model_dir
        else:
            os.makedirs(data_source_dir, exist_ok=True)
    _ensure_dir_link(data_source_dir, os.path.join(TABDIFF_DIR, "data", hashed))
    logger.info("Linked data: %s -> %s", os.path.join(TABDIFF_DIR, "data", hashed), data_source_dir)

    ckpt_source_dir = os.path.join(abs_model_dir, "ckpt")
    if not os.path.isdir(ckpt_source_dir):
        try:
            has_pt = any(f.endswith(".pt") for f in os.listdir(abs_model_dir))
        except Exception:
            has_pt = False
        ckpt_source_dir = abs_model_dir if has_pt else ckpt_source_dir
        os.makedirs(ckpt_source_dir, exist_ok=True)
    _ensure_dir_link(ckpt_source_dir, os.path.join(TABDIFF_DIR, "tabdiff", "ckpt", hashed))
    logger.info("Linked ckpt: %s -> %s", os.path.join(TABDIFF_DIR, "tabdiff", "ckpt", hashed), ckpt_source_dir)
    return hashed


def _find_ckpt_in(abs_model_dir: str):
    ckpt_source_dir = os.path.join(abs_model_dir, "ckpt")
    if not os.path.isdir(ckpt_source_dir):
        ckpt_source_dir = abs_model_dir
    for d in (os.path.join(ckpt_source_dir, "learnable_schedule"), ckpt_source_dir):
        if not os.path.isdir(d):
            continue
        ckpts = [f for f in os.listdir(d) if f.endswith(".pt") and "model" in f.lower()]
        if ckpts:
            best = ([f for f in ckpts if f.startswith("best_ema_model")]
                    or [f for f in ckpts if f.startswith("best_model")] or ckpts)
            ckpt_path = os.path.join(d, sorted(best)[-1])
            logger.info("Found best checkpoint in model_dir: %s", ckpt_path)
            return ckpt_path
    return None


# --- contract entry points --------------------------------------------------

def train(req: dict) -> Path:
    """Train iff no checkpoint exists; always leaves a contract meta.json.

    Decoupled from generate: trains TabDiff unconditionally (constraints applied
    later at inference) and records the column order for output reconciliation.
    """
    abs_model_dir = os.path.abspath(req["output_model_dir"])
    os.makedirs(abs_model_dir, exist_ok=True)
    dataset = req["dataset"]
    train_csv = req["train_csv"]
    label_col = req.get("label_column")
    device = _resolve_device(req.get("device"))
    gpu_index = 0 if device.startswith("cuda") else -1
    num_epochs = req.get("num_epochs")
    check_val_every = req.get("check_val_every")

    with _ModelDirLock(abs_model_dir):
        hashed = _link_model_dir(abs_model_dir, dataset)

        if not _tabdiff_has_checkpoint(hashed):
            # Link real/test CSV TabDiff reads for in-training evaluation.
            syn_data_dir = os.path.join(TABDIFF_DIR, "synthetic", hashed)
            os.makedirs(syn_data_dir, exist_ok=True)
            for fname in ("real.csv", "test.csv"):
                if train_csv and os.path.exists(train_csv):
                    _ensure_file_link(os.path.abspath(train_csv), os.path.join(syn_data_dir, fname))

            _train_tabdiff_model(hashed, train_csv=train_csv, label_col=label_col,
                                 gpu_index=gpu_index, num_epochs=num_epochs,
                                 check_val_every=check_val_every)
        else:
            logger.info("Found existing TabDiff checkpoint for %s, skipping training.", hashed)

        columns = list(pd.read_csv(train_csv, nrows=0).columns) if os.path.exists(train_csv) else []
        meta = {
            "schema_version": "1.0",
            "method": req["method"],
            "dataset": dataset,
            "label_column": label_col,
            "columns": columns,
            "trained_at": datetime.datetime.utcnow().isoformat() + "Z",
            "sdcontract_core_version": "1.0",
            # TabDiff-specific (consumed by generate):
            "dataname": hashed,
        }
        (Path(abs_model_dir) / META_NAME).write_text(json.dumps(meta, indent=2))
    return Path(abs_model_dir)


def generate(req: dict, native_constraints: list[str], privacy_mode: str = "none") -> Path:
    """Sample to the host-chosen output_csv_path under native (col=val)
    constraints; never retrains."""
    abs_model_dir = os.path.abspath(req["model_dir"])
    meta = json.loads((Path(abs_model_dir) / META_NAME).read_text())
    dataset = meta["dataset"]
    n_samples = int(req["n_samples"])
    device = _resolve_device(req.get("device"))
    num_inference_steps = req.get("num_inference_steps")

    hashed = _link_model_dir(abs_model_dir, dataset)
    ckpt_path = _find_ckpt_in(abs_model_dir)
    if ckpt_path is None and not _tabdiff_has_checkpoint(hashed):
        raise RuntimeError(
            f"No TabDiff checkpoint under {abs_model_dir}; train before generate "
            "(generate never retrains)."
        )

    # Map privacy mode -> sampling knobs (mirrors tabdiff_wrapper.py main()).
    if privacy_mode == "stochastic":
        s_churn, privacy_noise_scale, cat_noise_scale = _S_CHURN_DEFAULT, 0.0, 0.2
    elif privacy_mode == "midpoint":
        s_churn, privacy_noise_scale, cat_noise_scale = 0.0, _PRIVACY_NOISE_DEFAULT, 0.2
    else:  # none
        s_churn, privacy_noise_scale, cat_noise_scale = 0.0, 0.0, 0.0

    constraints = []
    if privacy_mode != "universal":
        for raw in native_constraints:
            if "=" in raw:
                k, v = raw.split("=", 1)
                constraints.append((k.strip(), v.strip()))

    # tabdiff-universal: full native dialect via guidance.
    if privacy_mode == "universal":
        if not native_constraints:
            df = generate_unconditional(dataname=hashed, num_samples=n_samples,
                                        ckpt_path=ckpt_path, device=device,
                                        num_inference_steps=num_inference_steps)
        else:
            df = generate_guided(
                dataname=hashed, constraint_specs=list(native_constraints),
                num_samples=n_samples, ckpt_path=ckpt_path, device=device,
                num_inference_steps=num_inference_steps,
                cat_snap_final=req.get("cat_snap_final", False),
            )
    else:
        # Unconditional only when there is genuinely nothing to condition on.
        use_unconditional = (privacy_mode == "none" and not constraints)
        if use_unconditional:
            df = generate_unconditional(dataname=hashed, num_samples=n_samples,
                                        ckpt_path=ckpt_path, device=device,
                                        num_inference_steps=num_inference_steps)
        else:
            df = generate_multi_conditional(
                dataname=hashed, constraints=constraints, num_samples=n_samples,
                s_churn=s_churn, privacy_noise_scale=privacy_noise_scale,
                cat_noise_scale=cat_noise_scale, impute_condition="x_t",
                ckpt_path=ckpt_path, device=device, num_inference_steps=num_inference_steps,
            )

    # Reconcile column names/order against the training schema recorded at train.
    actual_cols = meta.get("columns") or list(df.columns)
    rename_map = {}
    for c in df.columns:
        if c in actual_cols:
            continue
        alt_c = str(c).replace(".", "-")
        if alt_c in actual_cols:
            rename_map[c] = alt_c
        else:
            for real_c in actual_cols:
                if str(c).lower().replace(".", "") == str(real_c).lower().replace("-", ""):
                    rename_map[c] = real_c
                    break
    if rename_map:
        df.rename(columns=rename_map, inplace=True)
        logger.info("Renamed columns to match training schema: %s", rename_map)
    missing = set(actual_cols) - set(df.columns)
    if missing:
        raise RuntimeError(f"[tabdiff] Generated output is missing columns: {sorted(missing)}.")
    df = df[actual_cols]

    out_path = Path(req["output_csv_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info("Saved %d samples -> %s", len(df), out_path)
    return out_path
