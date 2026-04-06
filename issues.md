# Identified Issues and Architectural Risks

This document outlines critical issues identified in the current state of the repository that may lead to model failures, training errors, or broken portability.

## 1. Hardcoded External Paths (Broken Portability)
The script `compare_privacy.py` contains a hardcoded absolute path to an external repository on a specific local machine:
```python
SD_EVAL_ROOT = "/mnt/c/Users/gaura/Documents/GitHub/sd_eval"
```
This will cause immediate failures on any other system. Furthermore, many configuration files in `data/Info/` (e.g., `shoppers.json`, `adult.json`, `beijing_dcr.json`) point to data paths outside the repository (`../../data/processed/` or `../../data/processed/`). Since `download_dataset.py` does not populate these external directories, the data pipeline will fail for new users.

## 2. Brittle Checkpoint Selection
In `tabdiff/main.py`, the logic for loading checkpoints has been changed to a brittle pattern:
```python
ckpt_path_arr = glob.glob(f"{ckpt_parent_path}/*.pt")
ckpt_path = ckpt_path_arr[-1]
```
`glob.glob` returns files in an arbitrary order. Because the checkpoint directory contains various files (e.g., `best_model_...`, `best_ema_model_...`, and epoch-specific `model_...`), this logic can randomly pick a non-optimal or incorrect model. The previous implementation specifically targeted `best_ema_model*`.

## 3. Conflicting Categorical Encoding Logic
Recent changes to `process_dataset.py` introduced pre-encoding of categorical data into integers using `OrdinalEncoder` with `unknown_value=-1`. However, the core training pipeline in `src/data.py` also runs its own `OrdinalEncoder`.
*   The `-1` value for unknowns in the test set may be re-mapped by the second encoder to `max_values + 1`.
*   In `UniModMLP`, the value `max_values + 1` is often reserved for the **mask category**. This collision could cause the model to incorrectly treat unknown categorical values as masked tokens during inference.

## 4. Risk of Mixed-Type Data Arrays
In `process_dataset.py`, categorical features are encoded into integers before being saved to `.npy`, but the target column (`y_train`) is often saved as original strings. In `utils_train.py`, these are concatenated:
```python
return np.concatenate([y.reshape(-1, 1), X], axis=1) # y (strings) + X (ints)
```
This forces `numpy` to cast the entire feature array to strings (the common denominator), which are then passed to the model's transformation pipeline. This is inefficient and risks unexpected behavior during the re-encoding process in `src/data.py`.

## 5. Misleading "Quick" Generation Script
The new `quick_conditional_generate.py` script is advertised as using "TabDiff's infrastructure" but bypasses the diffusion model entirely. It simply performs noise-augmented sampling from the training data. This could lead to incorrect conclusions when evaluating the model's actual conditional generation capabilities.

## 6. Reduced Training Budget
The default training steps in `tabdiff/configs/tabdiff_configs.toml` were reduced from 8000 to 4000. While this reduces training time, it may lead to performance regressions or under-trained models for complex datasets.
