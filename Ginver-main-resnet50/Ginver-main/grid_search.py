import subprocess
import os
import itertools
import pandas as pd
import re
from datetime import datetime
import sys

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
    
    return mse_loss

def params_to_key(params, comparison_keys=None):
    """
    Convert a parameter dictionary to a string key for comparison
    Only uses keys that are present in comparison_keys if provided
    """
    if comparison_keys:
        # Only include keys that are in both params and comparison_keys
        filtered_params = {k: params[k] for k in comparison_keys if k in params}
    else:
        filtered_params = {k: v for k, v in params.items() if k != 'mse_loss'}
    
    return "_".join([f"{k}={v}" for k, v in sorted(filtered_params.items())])

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

def grid_search():
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
        'layer': [2],
        'batch-size': [64],
        'lr': [0.0002],
        'tv-weight': [0.01, 0.025, 0.05, 0.1, 0.5], # it is not been used
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
    
    # Initialize results DataFrame if no existing results
    if existing_results is None:
        results = []
        df = pd.DataFrame(columns=csv_columns)
    else:
        results = existing_results.to_dict('records')
        df = existing_results
    
    # Execute grid search
    total_combinations = len(combinations)
    executed_combinations = 0
    skipped_combinations = 0
    
    for i, values in enumerate(combinations):
        params = dict(zip(keys, values))
        
        # Check if this combination has already been tested
        if existing_results is not None and has_been_tested(existing_results, params):
            print(f"\nSkipping combination {i+1}/{total_combinations} - already tested:")
            print(params)
            skipped_combinations += 1
            continue
        
        print(f"\nCombination {i+1}/{total_combinations}:")
        print(params)
        
        # Execute attack.py with current parameters
        mse_loss = run_attack(params)
        executed_combinations += 1
        
        # Create a standardized result dictionary with only specific columns
        result_dict = {}
        for col in csv_columns:
            if col == 'mse_loss':
                result_dict[col] = mse_loss
            elif col in params:
                result_dict[col] = params[col]
            else:
                # Use default value if parameter not specified
                result_dict[col] = default_values.get(col, None)
        
        results.append(result_dict)
        
        # Update and save results after each execution
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
    
    print(f"\nGrid search completed: {executed_combinations} combinations executed, {skipped_combinations} combinations skipped.")
    
    # Final results analysis only if we have results
    if len(results) > 0 and any(pd.notna(df['mse_loss'])):
        best_idx = df['mse_loss'].idxmin()
        best_params = df.loc[best_idx]
        
        print("\n" + "="*50)
        print("GRID SEARCH RESULTS")
        print("="*50)
        print(f"Best MSE Loss: {best_params['mse_loss']}")
        print("Best hyperparameters:")
        for key in csv_columns:
            if key != 'mse_loss':
                print(f"  {key}: {best_params[key]}")

        # Save final results sorted by MSE loss with only specified columns
        df.sort_values('mse_loss')[csv_columns].to_csv(final_results_file, index=False)
        
        return best_params
    else:
        print("\nNo valid results found. Check for errors in the attack script execution.")
        return None

if __name__ == "__main__":
    # Create results directory if it doesn't exist
    os.makedirs('../grid_search_results', exist_ok=True)
    
    # Execute grid search
    best_params = grid_search()
    
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