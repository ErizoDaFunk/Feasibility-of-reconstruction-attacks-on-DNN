import subprocess
import os
import itertools
import pandas as pd
import re
import concurrent.futures
from datetime import datetime
import sys
import numpy as np

# Import default parameters from attack.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from attack import get_default_params

def run_attack(params):
    """Executes attack.py with the specified parameters and returns the best MSE loss"""
    cmd = ["python", "attack.py"]
    
    # Add parameters to the command
    for param, value in params.items():
        if param in ['no-cuda', 'save-model'] and value is True:
            # For boolean arguments, only add the flag without a value
            cmd.append(f"--{param}")
        else:
            # For other arguments, add both the name and value
            cmd.append(f"--{param}")
            cmd.append(str(value))
    
    # Execute the command
    print(f"Executing: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Check for errors
    if result.returncode != 0:
        print(f"Error executing command: {result.stderr}")
        return None
    
    # Extract the best MSE loss using regular expressions
    output = result.stdout
    early_stopping_match = re.search(r"Early stopping triggered.*Best MSE loss: (\d+\.\d+)", output)
    
    # Fix: Convert list to string before applying regex
    last_lines = "\n".join(output.splitlines()[-10:])
    final_epoch_match = re.search(r"Epoch: \d+ Average MSE loss: (\d+\.\d+)", last_lines)
    
    if early_stopping_match:
        mse_loss = float(early_stopping_match.group(1))
    elif final_epoch_match:
        mse_loss = float(final_epoch_match.group(1))
    else:
        print("Could not extract MSE loss from output")
        print("Last 10 lines of output:")
        print(last_lines)
        return None
    
    # Return both the parameters and the MSE loss
    return {**params, 'mse_loss': mse_loss}

def has_been_tested(existing_results, params):
    """Check if this parameter combination has already been tested"""
    if existing_results is None or len(existing_results) == 0:
        return False
    
    # Get common keys between params and existing_results that we want to compare
    # Consider only these important parameters for comparison
    important_keys = ['layer', 'batch-size', 'tv-weight', 'patience', 'lr']
    common_keys = [k for k in important_keys if k in params and k in existing_results.columns]
    
    # Verify we have enough keys to make a meaningful comparison
    if len(common_keys) < 3:  # At least need layer, batch-size, and tv-weight
        print(f"Not enough common keys to compare: {common_keys}")
        return False
    
    # For each row in existing results, check if it matches current params
    for _, row in existing_results.iterrows():
        match = True
        for key in common_keys:
            # Convert to float for numerical comparison to avoid int/float mismatch
            param_val = float(params[key])
            row_val = float(row[key])
            
            # Use small tolerance for float comparison
            if abs(param_val - row_val) > 1e-6:
                match = False
                break
        
        if match:
            print(f"Found matching parameter set in existing results")
            return True
    
    return False

def parallel_grid_search(max_workers=None):
    """
    Execute grid search in parallel using ThreadPoolExecutor
    
    Parameters:
        max_workers (int): Maximum number of worker threads to use.
                          If None, it will use the default (typically CPU count)
    """
    # Define results directory and fixed results filename
    results_dir = '../grid_search_results'
    os.makedirs(results_dir, exist_ok=True)
    
    results_file = os.path.join(results_dir, 'grid_search_results.csv')
    final_results_file = os.path.join(results_dir, 'grid_search_final_results.csv')

    # Define columns that should be saved in the CSV files
    # Only include parameters we want to track in the results
    csv_columns = ['layer', 'batch-size', 'test-batch-size', 'epochs', 'tv-weight', 'patience', 'lr', 'gamma', 'mse_loss']
    
    # Get default values from attack.py
    default_values = get_default_params()
    
    # Define hyperparameter values to explore
    param_grid = {
        'mode': ['whitebox'],
        'layer': [2],
        'batch-size': [64],
        'lr': [0.0002, 0.0001],  # More values
        'tv-weight': [0.8],  # More values
        'patience': [2],
        'epochs': [14],  # Fixed epochs since we have early stopping
        'no-cuda': [True],  # Use CPU for testing
        'save-model': [True]
    }
    
    # Create all possible parameter combinations
    keys = list(param_grid.keys())
    combinations = list(itertools.product(*[param_grid[key] for key in keys]))
    
    # Load existing results if file exists
    existing_results = None
    if os.path.exists(results_file):
        try:
            existing_results = pd.read_csv(results_file)
            print(f"Loaded existing results from {results_file} with {len(existing_results)} entries.")
        except Exception as e:
            print(f"Error loading existing results: {e}")
            existing_results = None
    
    # Initialize results list
    if existing_results is None:
        results = []
    else:
        results = existing_results.to_dict('records')
    
    # Filter out combinations that have already been tested
    combinations_to_run = []
    for values in combinations:
        params = dict(zip(keys, values))
        if existing_results is None or not has_been_tested(existing_results, params):
            combinations_to_run.append(params)
    
    total_combinations = len(combinations)
    combinations_to_run_count = len(combinations_to_run)
    print(f"Total combinations: {total_combinations}")
    print(f"Combinations to run: {combinations_to_run_count}")
    print(f"Combinations already tested: {total_combinations - combinations_to_run_count}")
    
    # Run grid search in parallel
    if combinations_to_run_count > 0:
        # Determine number of workers - use a reasonable number based on CPUs
        if max_workers is None:
            # Use number of CPUs available, but never more than 48
            max_workers = min(48, os.cpu_count() or 1)
        
        print(f"Running grid search with {max_workers} workers")
        
        # Execute grid search in parallel
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks to the executor
            future_to_params = {executor.submit(run_attack, params): params for params in combinations_to_run}
            
            # Process results as they complete
            for i, future in enumerate(concurrent.futures.as_completed(future_to_params)):
                params = future_to_params[future]
                try:
                    result = future.result()
                    if result is not None:
                        results.append(result)
                        print(f"Completed combination {i+1}/{combinations_to_run_count}")
                        print(f"Parameters: {params}")
                        print(f"MSE Loss: {result['mse_loss']}")
                        
                        # Update and save results after each completion
                        df = pd.DataFrame(results)
                        # Ensure all specified columns exist
                        for col in csv_columns:
                            if col not in df.columns:
                                df[col] = default_values.get(col, None)
                        
                        # Save only the specified columns
                        df[csv_columns].to_csv(results_file, index=False)
                        
                        # Show best result so far
                        if len(results) > 0 and any(pd.notna(df['mse_loss'])):
                            best_idx = df['mse_loss'].idxmin()
                            if pd.notna(best_idx):
                                print(f"\nBest result so far:")
                                print(f"MSE Loss: {df.loc[best_idx, 'mse_loss']}")
                                print(f"Parameters: {df.loc[best_idx, csv_columns].to_dict()}")
                    else:
                        print(f"Combination {i+1}/{combinations_to_run_count} failed")
                        print(f"Parameters: {params}")
                except Exception as e:
                    print(f"Exception occurred during execution: {e}")
                    print(f"Parameters: {params}")
    
    # Final analysis
    if os.path.exists(results_file):
        df = pd.read_csv(results_file)
        if len(df) > 0 and any(pd.notna(df['mse_loss'])):
            # Sort by MSE loss and save final results
            df = df.sort_values('mse_loss')
            df[csv_columns].to_csv(final_results_file, index=False)
            
            # Get best parameters
            best_params = df.iloc[0].to_dict()
            
            print("\n" + "="*50)
            print("GRID SEARCH RESULTS")
            print("="*50)
            print(f"Best MSE Loss: {best_params['mse_loss']}")
            print("Best hyperparameters:")
            for key in csv_columns:
                if key != 'mse_loss':
                    print(f"  {key}: {best_params[key]}")
                    
            return best_params
        else:
            print("\nNo valid results found.")
            return None
    else:
        print("\nNo results file found.")
        return None

if __name__ == "__main__":
    # Create results directory if it doesn't exist
    os.makedirs('../grid_search_results', exist_ok=True)
    
    # Execute parallel grid search with 40 workers (adjust as needed for your system)
    # For a 48 CPU system, using 40-45 workers is a good balance
    best_params = parallel_grid_search(max_workers=2)
    
    # Command to execute the model with best parameters
    if best_params is not None:
        best_cmd = ["python", "attack.py"]
        for param, value in best_params.items():
            if param != 'mse_loss':  # Exclude MSE loss
                if param in ['no-cuda', 'save-model'] and value is True:
                    best_cmd.append(f"--{param}")
                else:
                    best_cmd.append(f"--{param}")
                    best_cmd.append(str(value))
        
        print("\nCommand to run the best model:")
        print(" ".join(best_cmd))