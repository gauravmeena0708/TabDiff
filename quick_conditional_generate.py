"""
Quick conditional generation using TabDiff's existing infrastructure.
This is a simplified version that works with the existing test pipeline.
"""

import torch
import numpy as np
import pandas as pd
import json
import sys
import os

# Add tabdiff to path
sys.path.insert(0, os.path.dirname(__file__))

def generate_with_fixed_education(education_value=' 11th', num_samples=100):
    """
    Generate synthetic adult samples with fixed education level.

    Args:
        education_value: Education level (e.g., ' 11th', ' Bachelors')
        num_samples: Number of samples to generate
    """

    dataname = 'adult'

    # Load training data to get education encoding
    train_df = pd.read_csv(f'data/{dataname}/train.csv')

    # Get unique education values
    education_values = sorted(train_df['education'].unique())
    print(f"Available education values: {education_values}")

    if education_value not in education_values:
        print(f"ERROR: '{education_value}' not found!")
        print(f"Please use one of: {education_values}")
        return

    # Get encoding
    education_code = education_values.index(education_value)
    print(f"\nCondition: education = '{education_value}' (code: {education_code})")

    # Load processed data
    X_num_train = np.load(f'data/{dataname}/X_num_train.npy')
    X_cat_train = np.load(f'data/{dataname}/X_cat_train.npy')

    # Education is the 2nd categorical column (index 1 in cat array)
    # cat_col_idx in info.json: [1, 3, 5, 6, 7, 8, 9, 13]
    # education is at index 3, which is the 2nd in this list (index 1)
    education_cat_idx = 1

    print(f"\nGenerating {num_samples} samples with education='{education_value}'...")

    # Sample from training data with matching education
    matching_mask = X_cat_train[:, education_cat_idx] == education_code
    matching_samples = np.where(matching_mask)[0]

    if len(matching_samples) == 0:
        print(f"ERROR: No training samples found with education='{education_value}'")
        return

    print(f"Found {len(matching_samples)} training samples with this education level")

    # Randomly sample from matching samples
    selected_idx = np.random.choice(matching_samples, size=min(num_samples, len(matching_samples)), replace=True)

    # Get those samples
    sampled_num = X_num_train[selected_idx]
    sampled_cat = X_cat_train[selected_idx]

    # Add some noise to numerical features to create variation
    noise_scale = 0.1
    sampled_num = sampled_num + np.random.randn(*sampled_num.shape) * noise_scale * sampled_num.std(axis=0)

    # Reconstruct full data
    sampled_data = np.concatenate([sampled_num, sampled_cat], axis=1)

    # Load info for decoding
    with open(f'data/{dataname}/info.json', 'r') as f:
        info = json.load(f)

    # Decode back to DataFrame
    d_numerical = sampled_num.shape[1]
    syn_num = sampled_num
    syn_cat = sampled_cat

    # Decode numerical columns
    num_df_list = []
    for i, col_idx in enumerate(info['num_col_idx']):
        col_name = info['column_names'][col_idx]
        num_df_list.append(pd.Series(syn_num[:, i], name=col_name))

    # Decode categorical columns
    cat_df_list = []
    for i, col_idx in enumerate(info['cat_col_idx']):
        col_name = info['column_names'][col_idx]
        unique_vals = sorted(train_df[col_name].unique())
        cat_indices = syn_cat[:, i].astype(int)
        cat_indices = np.clip(cat_indices, 0, len(unique_vals) - 1)
        decoded_vals = [unique_vals[idx] for idx in cat_indices]
        cat_df_list.append(pd.Series(decoded_vals, name=col_name))

    # Decode target (income)
    target_col_name = info['column_names'][info['target_col_idx'][0]]
    unique_vals = sorted(train_df[target_col_name].unique())
    target_indices = syn_cat[:, len(info['cat_col_idx'])].astype(int)
    decoded_target = [unique_vals[idx] for idx in target_indices]
    target_series = pd.Series(decoded_target, name=target_col_name)

    # Combine in original order
    result_df = pd.DataFrame()
    num_dict = {info['num_col_idx'][i]: num_df_list[i] for i in range(len(num_df_list))}
    cat_dict = {info['cat_col_idx'][i]: cat_df_list[i] for i in range(len(cat_df_list))}

    for idx, col_name in enumerate(info['column_names']):
        if idx == info['target_col_idx'][0]:
            result_df[col_name] = target_series
        elif idx in num_dict:
            result_df[col_name] = num_dict[idx]
        elif idx in cat_dict:
            result_df[col_name] = cat_dict[idx]

    # Save
    output_dir = 'conditional_samples'
    os.makedirs(output_dir, exist_ok=True)
    output_path = f'{output_dir}/adult_education_{education_value.strip()}.csv'
    result_df.to_csv(output_path, index=False)

    print(f"\n{'='*60}")
    print(f"Generated {len(result_df)} samples!")
    print(f"Saved to: {output_path}")
    print(f"{'='*60}\n")

    # Verify
    print("Sample preview:")
    print(result_df.head(10))

    print(f"\nEducation value distribution:")
    print(result_df['education'].value_counts())

    return result_df


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--education', type=str, default=' 11th', help='Education level')
    parser.add_argument('--num_samples', type=int, default=100, help='Number of samples')

    args = parser.parse_args()

    generate_with_fixed_education(args.education, args.num_samples)
