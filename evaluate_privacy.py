
import pandas as pd
import json
import torch
import os
from tabdiff.metrics import TabMetrics

def evaluate_privacy_metrics(dataname='adult', condition_column='education', condition_value=' 11th'):
    print(f"\n--- Evaluating Privacy Metrics for {dataname} ({condition_column}={condition_value}) ---")
    
    # Paths
    real_data_path = f'data/{dataname}/train.csv'
    test_data_path = f'data/{dataname}/test.csv'
    val_data_path = None # Adult doesn't seem to have a val.csv in the info.json
    info_path = f'data/{dataname}/info.json'
    
    with open(info_path, 'r') as f:
        info = json.load(f)
        
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Initialize Metrics
    metrics_evaluator = TabMetrics(
        real_data_path=real_data_path,
        test_data_path=test_data_path,
        val_data_path=val_data_path,
        info=info,
        device=device,
        metric_list=['dcr']
    )
    
    methods = ['none', 'stochastic', 'midpoint']
    results = []
    
    for method in methods:
        file_path = f"privacy_comparison/{method}/{dataname}_{condition_column}_{condition_value}.csv"
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} not found.")
            continue
            
        print(f"Calculating DCR for {method}...")
        syn_df = pd.read_csv(file_path)
        
        # We need to make sure the columns match info['column_names'] order
        # But TabMetrics.evaluate_dcr seems to handle it via indexing if columns are integers,
        # or it uses column names. Let's check tabdiff/metrics.py again.
        # Actually, evaluate_dcr uses info['num_col_idx'] etc. and indexes into the df.
        
        out_metrics, out_extras = metrics_evaluator.evaluate_dcr(syn_df)
        
        results.append({
            "method": method,
            "dcr_score": out_metrics['dcr']
        })
        
    return pd.DataFrame(results)

if __name__ == "__main__":
    df_results = evaluate_privacy_metrics()
    print("\n--- Privacy Metric Results (DCR) ---")
    print("DCR Score: Higher means more potential memorization of training data.")
    print("A score around 0.5 is ideal (generated record is as likely to be closer to test as to train).")
    print(df_results.to_string(index=False))
    
    # Also evaluate the Bachelors condition
    df_results_bachelors = evaluate_privacy_metrics(condition_value=' Bachelors')
    print("\n--- Privacy Metric Results (DCR) - Bachelors ---")
    print(df_results_bachelors.to_string(index=False))
