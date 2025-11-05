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
from tabdiff.modules.main_modules import MLPDiffusion
import argparse

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
    categories = X_cat_train.max(axis=0).astype(int) + 1

    # Find checkpoint if not provided
    if ckpt_path is None:
        exp_dir = f'exp/{dataname}/learnable_schedule'
        if os.path.exists(exp_dir):
            ckpts = [f for f in os.listdir(exp_dir) if f.startswith('model_') and f.endswith('.pt')]
            if ckpts:
                latest_ckpt = sorted(ckpts, key=lambda x: int(x.split('_')[1].split('.')[0]))[-1]
                ckpt_path = os.path.join(exp_dir, latest_ckpt)
                print(f"Using checkpoint: {ckpt_path}")
            else:
                raise FileNotFoundError(f"No checkpoint found in {exp_dir}")
        else:
            raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")

    # Load checkpoint
    checkpoint = torch.load(ckpt_path, map_location=device)

    # Create model (you may need to adjust these parameters based on your training config)
    denoise_fn = MLPDiffusion(
        d_in=d_numerical + len(categories),
        num_classes=categories.tolist(),
        is_y_cond=False,
        rtdl_params={
            'd_layers': [512, 512, 512],
            'dropout': 0.0
        }
    ).to(device)

    # Load model weights
    denoise_fn.load_state_dict(checkpoint['model_state_dict'])

    # Create diffusion model
    diffusion = UnifiedCTimeDiffusion(
        num_classes=categories.tolist(),
        denoise_fn=denoise_fn,
        num_numerical_features=d_numerical,
        device=device,
        num_timesteps=1000,
        gaussian_loss_type='mse',
        scheduler='cosine',
        is_fair=False,
        fix_schedule=False
    ).to(device)

    # Load schedule parameters
    diffusion.num_schedule.load_state_dict(checkpoint['num_schedule_state_dict'])
    diffusion.cat_schedule.load_state_dict(checkpoint['cat_schedule_state_dict'])

    diffusion.eval()

    return diffusion, info, X_num_train, X_cat_train, d_numerical, categories


def get_category_encoding(dataname, column_name, value):
    """Get the encoded value for a categorical column"""

    # Load the info.json with category mappings
    info_path = f'data/{dataname}/info.json'
    with open(info_path, 'r') as f:
        info = json.load(f)

    # Get column index
    col_idx = info['column_names'].index(column_name)

    # Load training data to see the mapping
    train_df = pd.read_csv(f'data/{dataname}/train.csv')

    # Get unique values and create mapping
    unique_vals = sorted(train_df[column_name].unique())

    if value not in unique_vals:
        print(f"Warning: '{value}' not found in training data for column '{column_name}'")
        print(f"Available values: {unique_vals}")
        # Use the first value as fallback
        value = unique_vals[0]
        print(f"Using fallback: '{value}'")

    encoded_value = unique_vals.index(value)

    print(f"\nColumn: {column_name}")
    print(f"Original value: {value}")
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
    diffusion, info, X_num_train, X_cat_train, d_numerical, categories = load_model_and_info(
        dataname, device=device
    )

    # Get encoding for the condition
    encoded_value, col_idx, unique_vals = get_category_encoding(
        dataname, condition_column, condition_value
    )

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
        cat_position = info['cat_col_idx'].index(col_idx)
        x_cat[:, cat_position] = encoded_value
        print(f"\nSetting categorical column {condition_column} (position {cat_position}) to {condition_value} (encoded: {encoded_value})")
    else:
        num_position = info['num_col_idx'].index(col_idx)
        # You'll need to set this to the appropriate normalized value
        x_num[:, num_position] = 0.0  # Placeholder - adjust based on your normalization
        print(f"\nSetting numerical column {condition_column} (position {num_position}) to {condition_value}")

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
    syn_df = decode_synthetic_data(syn_data, info, dataname, d_numerical)

    # Verify the condition column has the right value
    actual_values = syn_df[condition_column].unique()
    print(f"\nVerification - Unique values in {condition_column}: {actual_values}")

    return syn_df


def decode_synthetic_data(syn_data, info, dataname, d_numerical):
    """Decode synthetic data back to original format"""

    # Load the preprocessing info
    data_dir = f'data/{dataname}'
    info_json_path = f'{data_dir}/info.json'
    with open(info_json_path, 'r') as f:
        info_detail = json.load(f)

    # Split numerical and categorical
    syn_num = syn_data[:, :d_numerical].numpy()
    syn_cat = syn_data[:, d_numerical:].numpy()

    # Load training data to get categories
    train_df = pd.read_csv(f'{data_dir}/train.csv')

    # Decode numerical columns
    num_cols = [info['column_names'][i] for i in info['num_col_idx']]
    num_df = pd.DataFrame(syn_num, columns=[f'num_{i}' for i in range(syn_num.shape[1])])

    # Decode categorical columns
    cat_cols = [info['column_names'][i] for i in info['cat_col_idx']]
    cat_dfs = []
    for idx, col_name in enumerate(cat_cols):
        unique_vals = info['cat_encoders'][col_name] # Use stored categories
        cat_indices = syn_cat[:, idx].astype(int)
        # Clip to valid range
        cat_indices = np.clip(cat_indices, 0, len(unique_vals) - 1)
        decoded_vals = [unique_vals[i] for i in cat_indices]
        cat_dfs.append(pd.Series(decoded_vals, name=col_name))

    # Decode target column
    target_col_name = info['column_names'][info['target_col_idx'][0]]
    if info['task_type'] == 'binclass':
        # Target is categorical, already in syn_cat
        target_idx = len(cat_cols)
        unique_vals = info['cat_encoders'][target_col_name] # Use stored categories
        target_indices = syn_cat[:, target_idx].astype(int)
        target_indices = np.clip(target_indices, 0, len(unique_vals) - 1)
        decoded_target = [unique_vals[i] for i in target_indices]
        target_series = pd.Series(decoded_target, name=target_col_name)
    else:
        # Target is numerical
        target_series = pd.Series(syn_num[:, 0], name=target_col_name)
        syn_num = syn_num[:, 1:]
        num_cols = num_cols[1:]

    # Combine into final dataframe in original column order
    result_df = pd.DataFrame()

    num_col_map = {info['num_col_idx'][i]: (num_cols[i], syn_num[:, i])
                   for i in range(len(num_cols))}
    cat_col_map = {info['cat_col_idx'][i]: cat_dfs[i]
                   for i in range(len(cat_cols))}
    target_col_idx = info['target_col_idx'][0]

    for idx, col_name in enumerate(info['column_names']):
        if idx == target_col_idx:
            result_df[col_name] = target_series
        elif idx in num_col_map:
            result_df[col_name] = num_col_map[idx][1]
        elif idx in cat_col_map:
            result_df[col_name] = cat_col_map[idx]

    return result_df


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
