
import subprocess
import os
import pandas as pd
import json

def run_experiment(dataname, column, value, method, num_samples=100):
    print(f"\n>>> Running: Dataset={dataname}, Condition={column}={value}, Method={method}")
    
    cmd = [
        "python", "generate_conditional.py",
        "--dataname", dataname,
        "--condition_column", column,
        "--condition_value", str(value),
        "--num_samples", str(num_samples),
        "--privacy_method", method,
        "--output_dir", f"privacy_comparison/{method}"
    ]
    
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running experiment: {e}")
        return False

def compare_results(experiments):
    results = []
    
    for dataname, column, value in experiments:
        for method in ['none', 'stochastic', 'midpoint']:
            file_path = f"privacy_comparison/{method}/{dataname}_{column}_{value}.csv"
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                # Simple stats
                results.append({
                    "dataset": dataname,
                    "condition": f"{column}={value}",
                    "method": method,
                    "num_samples": len(df),
                    "unique_values_in_condition": df[column].nunique(),
                    "most_frequent_condition": df[column].mode()[0] if not df[column].empty else "N/A"
                })
    
    return pd.DataFrame(results)

if __name__ == "__main__":
    # Define experiments based on available datasets and models
    # Note: We assume models for these datasets are already trained.
    experiments = [
        ("adult", "education", " 11th"),
        ("adult", "education", " Bachelors"),
        ("adult", "occupation", " Sales")
    ]
    
    # Check what else is available in data/ and tabdiff/ckpt/
    # If other models are trained, they can be added here.
    
    os.makedirs("privacy_comparison", exist_ok=True)
    
    for dataname, column, value in experiments:
        for method in ['none', 'stochastic', 'midpoint']:
            run_experiment(dataname, column, value, method)
            
    summary_df = compare_results(experiments)
    print("\n--- Privacy Comparison Summary ---")
    print(summary_df.to_string(index=False))
    
    summary_df.to_csv("privacy_comparison/summary.csv", index=False)
    print(f"\nSummary saved to privacy_comparison/summary.csv")
