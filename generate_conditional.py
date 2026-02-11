"""
Generate synthetic data with specific column constraints.
Example: Generate samples where education='11th'
"""

import torch
import numpy as np
import pandas as pd
import json
import os
from tabdiff.models.unified_ctime_diffusion import UnifiedCtimeDiffusion
from tabdiff.modules.main_modules import MLPDiffusion, UniModMLP, Model
from utils_train import preprocess
from tabdiff.trainer import split_num_cat_target, recover_data
import argparse
import pickle

def load_model_and_info(dataname, ckpt_path=None, device='cuda'):
    """Load the trained TabDiff model and dataset info"""

    # Load dataset info
    info_path = f'data/{dataname}/info.json'
    with open(info_path, 'r') as f:
        info = json.load(f)

    # Load processed data to get categories and normalization info
    data_dir = f'data/{dataname}'
    X_num_train = np.load(f'{data_dir}/X_num_train.npy', allow_pickle=True)
    X_cat_train = np.load(f'{data_dir}/X_cat_train.npy', allow_pickle=True)

    # Get dimensions
    d_numerical = X_num_train.shape[1]
    # Find checkpoint if not provided
    if ckpt_path is None:
        # Try both possible directory structures
        possible_dirs = [
            f'tabdiff/ckpt/{dataname}/learnable_schedule',
            f'exp/{dataname}/learnable_schedule',
            f'tabdiff/ckpt/{dataname}/default_anonymized_retrain'
        ]

        exp_dir = None
        for dir_path in possible_dirs:
            if os.path.exists(dir_path):
                exp_dir = dir_path
                break
        
        # Fallback
        if exp_dir is None and os.path.exists(f'tabdiff/ckpt/{dataname}'):
            subdirs = [d for d in os.listdir(f'tabdiff/ckpt/{dataname}') if os.path.isdir(os.path.join(f'tabdiff/ckpt/{dataname}', d))]
            if subdirs:
                subdirs.sort(key=lambda x: os.path.getmtime(os.path.join(f'tabdiff/ckpt/{dataname}', x)), reverse=True)
                exp_dir = os.path.join(f'tabdiff/ckpt/{dataname}', subdirs[0])
                print(f"Using auto-detected experiment directory: {exp_dir}")

        if exp_dir is None:
            raise FileNotFoundError(f"Experiment directory not found. Tried: {possible_dirs}")

        # Look for checkpoint files with various naming patterns
        ckpts = [f for f in os.listdir(exp_dir) if f.endswith('.pt') and 'model' in f.lower()]

        if not ckpts:
            raise FileNotFoundError(f"No checkpoint found in {exp_dir}")

        # Prefer best_model or best_ema_model over regular model checkpoints
        best_models = [f for f in ckpts if f.startswith('best_model')]
        best_ema_models = [f for f in ckpts if f.startswith('best_ema_model')]

        if best_models:
            latest_ckpt = sorted(best_models)[-1]
        elif best_ema_models:
            latest_ckpt = sorted(best_ema_models)[-1]
        else:
            latest_ckpt = sorted(ckpts)[-1]

        ckpt_path = os.path.join(exp_dir, latest_ckpt)
        print(f"Using checkpoint: {ckpt_path}")

    # Load checkpoint and config
    checkpoint = torch.load(ckpt_path, map_location=device)

    # Load config from the same directory
    config_path = os.path.join(os.path.dirname(ckpt_path), 'config.pkl')
    with open(config_path, 'rb') as f:
        config = pickle.load(f)

    # Extract model parameters from config
    model_params = config['unimodmlp_params']
    diffusion_params = config['diffusion_params']
    edm_params = diffusion_params['edm_params']

    # Use categories from the config (includes target column)
    categories_with_mask = np.array(model_params['categories'])
    num_classes = (categories_with_mask - 1).astype(int)

    # Create model using the saved configuration
    unimodmlp = UniModMLP(
        d_numerical=model_params['d_numerical'],
        categories=model_params['categories'],
        num_layers=model_params['num_layers'],
        d_token=model_params['d_token'],
        n_head=model_params['n_head'],
        factor=model_params['factor'],
        bias=model_params['bias'],
        dim_t=model_params['dim_t'],
        use_mlp=model_params['use_mlp']
    ).to(device)

    # Wrap with EDM preconditioner (Model class)
    denoise_fn = Model(
        unimodmlp,
        **edm_params
    ).to(device)

    # Create diffusion model
    diffusion = UnifiedCtimeDiffusion(
        num_classes=num_classes,
        num_numerical_features=d_numerical,
        denoise_fn=denoise_fn,
        y_only_model=None,
        num_timesteps=diffusion_params['num_timesteps'],
        scheduler=diffusion_params['scheduler'],
        cat_scheduler=diffusion_params['cat_scheduler'],
        noise_dist=diffusion_params.get('noise_dist', 'uniform'),
        edm_params=edm_params,
        noise_dist_params=diffusion_params.get('noise_dist_params', {}),
        noise_schedule_params=diffusion_params.get('noise_schedule_params', {}),
        sampler_params=diffusion_params.get('sampler_params', {}),
        device=device
    ).to(device)

    # Load model weights from checkpoint
    diffusion._denoise_fn.load_state_dict(checkpoint['denoise_fn'])
    diffusion.num_schedule.load_state_dict(checkpoint['num_schedule'])
    diffusion.cat_schedule.load_state_dict(checkpoint['cat_schedule'])

    # Prepare inverse transforms to recover original feature scales
    data_params = config.get('data', {})
    dequant_dist = data_params.get('dequant_dist', 'none')
    int_dequant_factor = data_params.get('int_dequant_factor', 0)

    _, _, _, _, num_inverse, int_inverse, cat_inverse = preprocess(
        data_dir,
        y_only=False,
        dequant_dist=dequant_dist,
        int_dequant_factor=int_dequant_factor,
        task_type=info['task_type'],
        inverse=True
    )

    num_transform = getattr(num_inverse, '__self__', None) if hasattr(num_inverse, '__self__') else None
    if num_transform is not None and not hasattr(num_transform, 'transform'):
        num_transform = None
    int_transform = getattr(int_inverse, '__self__', None) if hasattr(int_inverse, '__self__') else None
    if int_transform is not None and not hasattr(int_transform, 'transform'):
        int_transform = None

    diffusion.eval()

    return (
        diffusion,
        info,
        X_num_train,
        X_cat_train,
        d_numerical,
        categories_with_mask,
        num_inverse,
        int_inverse,
        cat_inverse,
        num_transform,
        int_transform,
    )


def get_category_encoding(dataname, column_name, value):
    """Get the encoded value for a categorical column"""

    # Load the info.json with category mappings
    info_path = f'data/{dataname}/info.json'
    with open(info_path, 'r') as f:
        info = json.load(f)

    # Use cat_encoders from info.json nicely
    unique_vals = None
    if 'cat_encoders' in info and column_name in info['cat_encoders']:
        unique_vals = info['cat_encoders'][column_name]
    else:
        print(f"Warning: cat_encoders not found for {column_name}, falling back to train.csv")
        # Load training data only if needed
        train_df = pd.read_csv(f'data/{dataname}/train.csv')
        unique_vals = sorted(train_df[column_name].unique())

    # Get column index
    if column_name in info['column_names']:
        col_idx = info['column_names'].index(column_name)
    else:
        # Fallback for target column if not in column_names explicitly (some configs differ)
        col_idx = -1 
        # But generally it should be there.

    canonical_value = value
    matched_via_strip = False

    if canonical_value not in unique_vals:
        if isinstance(value, str):
            normalized_map = {}
            for raw in unique_vals:
                key = str(raw).strip()
                normalized_map.setdefault(key, raw)
            stripped_value = value.strip()
            if stripped_value in normalized_map:
                canonical_value = normalized_map[stripped_value]
                matched_via_strip = True
            else:
                print(f"Warning: '{value}' not found in categories for column '{column_name}'")
                print(f"Available values: {unique_vals}")
                try:
                    canonical_value = unique_vals[0]
                except:
                    canonical_value = value
                print(f"Using fallback: '{canonical_value}'")
        else:
            print(f"Warning: '{value}' not found in categories for column '{column_name}'")
            print(f"Available values: {unique_vals}")
            try:
                canonical_value = unique_vals[0]
            except:
                canonical_value = value
            print(f"Using fallback: '{canonical_value}'")

    try:
        encoded_value = unique_vals.index(canonical_value)
    except ValueError:
        encoded_value = 0
        print(f"Error: Could not encode value {canonical_value}. Defaulting to 0.")

    print(f"\nColumn: {column_name}")
    if matched_via_strip:
        print(f"Matched input '{value}' to dataset value '{canonical_value}' after stripping whitespace.")
    print(f"Original value: {canonical_value}")
    print(f"Encoded value: {encoded_value}")
    print(f"Available categories: {unique_vals}")

    return encoded_value, col_idx, unique_vals


def generate_conditional_samples(
    dataname='adult',
    condition_column='education',
    condition_value='11th',
    num_samples=100,
    w_num=0.6,
    w_cat=0.6,
    resample_rounds=1,
    ckpt_path=None,
    device='cuda'
):
    """
    Generate synthetic samples with a specific column fixed to a value.

    Args:
        dataname: Name of dataset
        condition_column: Name of column to condition on (e.g., 'education')
        condition_value: Value to fix the column to (e.g., '11th')
        num_samples: Number of samples to generate
        w_num: Guidance weight for numerical columns
        w_cat: Guidance weight for categorical columns
        resample_rounds: Number of resampling rounds per timestep
        device: Device to use
    """

    print(f"Loading model for {dataname}...")
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available. Falling back to CPU.")
        device = 'cpu'
    (
        diffusion,
        info,
        X_num_train,
        X_cat_train,
        d_numerical,
        categories,
        num_inverse,
        int_inverse,
        cat_inverse,
        num_transform,
        int_transform,
    ) = load_model_and_info(dataname, ckpt_path=ckpt_path, device=device)

    # Locate column metadata
    if condition_column not in info['column_names']:
        raise ValueError(f"Column '{condition_column}' not found in dataset.")
    col_idx = info['column_names'].index(condition_column)

    # Determine if condition column is numerical or categorical
    is_numerical = col_idx in info['num_col_idx']
    is_categorical = col_idx in info['cat_col_idx']

    if not (is_numerical or is_categorical):
        raise ValueError(f"Column {condition_column} is the target column. Use regular imputation instead.")

    # Create input tensors
    x_num = torch.randn(num_samples, d_numerical).to(device) * 0.5  # Start with noise
    x_cat = torch.zeros(num_samples, len(categories)).long().to(device)  # Start with category 0

    # Set the condition column
    if is_categorical:
        encoded_value, _, _ = get_category_encoding(
            dataname, condition_column, condition_value
        )
        cat_position = info['cat_col_idx'].index(col_idx)
        if info['task_type'] != 'regression':
            cat_position += 1  # offset for target column stored at position 0
        x_cat[:, cat_position] = encoded_value
        print(f"\nSetting categorical column {condition_column} (position {cat_position}) to {condition_value} (encoded: {encoded_value})")
    else:
        num_position = info['num_col_idx'].index(col_idx)
        try:
            numeric_value = float(condition_value)
        except (TypeError, ValueError):
            raise ValueError(f"Condition value for numerical column '{condition_column}' must be numeric.")

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
        print(
            f"\nSetting numerical column {condition_column} (position {num_position}) "
            f"to {condition_value} (normalized: {normalized_value:.4f})"
        )

    # Create masks: which columns to GENERATE (True = generate, False = keep fixed)
    # We want to generate ALL columns EXCEPT the condition column
    num_mask_idx = list(range(d_numerical))  # Generate all numerical columns
    cat_mask_idx = list(range(len(categories)))  # Generate all categorical columns

    # Remove the condition column from the mask
    if is_categorical:
        cat_mask_idx.remove(cat_position)
    else:
        num_mask_idx.remove(num_position)

    print(f"\nGenerating {num_samples} samples with {condition_column}='{condition_value}'...")
    print(f"Numerical columns to generate: {num_mask_idx}")
    print(f"Categorical columns to generate: {cat_mask_idx}")

    # Generate samples using the impute method
    # Note: This uses "x_0" conditioning to keep the condition column clean
    with torch.no_grad():
        syn_data = diffusion.sample_impute(
            x_num, x_cat,
            num_mask_idx, cat_mask_idx,
            resample_rounds=resample_rounds,
            impute_condition="x_0",  # Keep condition column clean
            w_num=w_num,
            w_cat=w_cat
        )

    print(f"\nGenerated samples shape: {syn_data.shape}")

    # Convert back to DataFrame
    syn_df = decode_synthetic_data(syn_data, info, num_inverse, int_inverse, cat_inverse)

    # Verify the condition column has the right value
    actual_values = syn_df[condition_column].unique()
    print(f"\nVerification - Unique values in {condition_column}: {actual_values}")

    return syn_df


def decode_synthetic_data(syn_data, info, num_inverse, int_inverse, cat_inverse):
    """Decode synthetic data back to original feature space using stored inverses."""

    syn_num, syn_cat, syn_target = split_num_cat_target(
        syn_data, info, num_inverse, int_inverse, cat_inverse
    )
    syn_df = recover_data(syn_num, syn_cat, syn_target, info)

    idx_name_mapping = info['idx_name_mapping']
    idx_name_mapping = {int(key): value for key, value in idx_name_mapping.items()}
    syn_df.rename(columns=idx_name_mapping, inplace=True)

    # Cast integer columns back to ints
    int_col_names = [info['column_names'][i] for i in info.get('int_col_idx', [])]
    for col_name in int_col_names:
        if col_name in syn_df.columns:
            numeric_series = pd.to_numeric(syn_df[col_name], errors='coerce').round()
            if numeric_series.isnull().any():
                syn_df[col_name] = numeric_series
            else:
                syn_df[col_name] = numeric_series.astype(int)

    # Map categorical indices back to string labels
    cat_indices = list(info.get('cat_col_idx', []))
    if info['task_type'] != 'regression':
        cat_indices += info.get('target_col_idx', [])

    for idx in cat_indices:
        col_name = info['column_names'][idx]
        if col_name not in syn_df.columns:
            continue
        # Use cat_encoders if available
        encoder_vals = None
        if 'cat_encoders' in info and col_name in info['cat_encoders']:
            encoder_vals = info['cat_encoders'][col_name]
        
        if encoder_vals is None:
            continue
            
        series = syn_df[col_name]
        series_num = pd.to_numeric(series, errors='coerce')
        if not series_num.isnull().all():
            indices = series_num.fillna(0).astype(int).clip(0, len(encoder_vals) - 1)
            mapped = indices.apply(lambda i: encoder_vals[i])
        else:
            mapped = series
        syn_df[col_name] = mapped.apply(lambda x: x.strip() if isinstance(x, str) else x)

    return syn_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate conditional synthetic data')
    parser.add_argument('--dataname', type=str, default='adult', help='Dataset name')
    parser.add_argument('--condition_column', type=str, default='education', help='Column to condition on')
    parser.add_argument('--condition_value', type=str, default='11th', help='Value to fix column to')
    parser.add_argument('--num_samples', type=int, default=100, help='Number of samples to generate')
    parser.add_argument('--w_num', type=float, default=0.0, help='Numerical guidance weight')
    parser.add_argument('--w_cat', type=float, default=0.0, help='Categorical guidance weight')
    parser.add_argument('--resample_rounds', type=int, default=1, help='Resampling rounds')
    parser.add_argument('--output_dir', type=str, default='conditional_samples', help='Output directory')
    parser.add_argument('--ckpt_path', type=str, default=None, help='Path to model checkpoint (optional)')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda or cpu)')

    args = parser.parse_args()

    # Generate samples
    syn_df = generate_conditional_samples(
        dataname=args.dataname,
        condition_column=args.condition_column,
        condition_value=args.condition_value,
        num_samples=args.num_samples,
        w_num=args.w_num,
        w_cat=args.w_cat,
        resample_rounds=args.resample_rounds,
        ckpt_path=args.ckpt_path,
        device=args.device
    )

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = f'{args.output_dir}/{args.dataname}_{args.condition_column}_{args.condition_value}.csv'
    syn_df.to_csv(output_path, index=False)

    print(f"\n{'='*60}")
    print(f"Successfully generated {len(syn_df)} samples!")
    print(f"Saved to: {output_path}")
    print(f"{'='*60}")

    # Show sample
    print("\nFirst 5 samples:")
    print(syn_df.head())

    # Show statistics for the condition column
    print(f"\nCondition column '{args.condition_column}' value distribution:")
    print(syn_df[args.condition_column].value_counts())
