import subprocess
import os
import itertools
import pandas as pd
import re
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import threading

# Add a lock for thread-safe printing and file operations
print_lock = threading.Lock()

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
    
    # Execute the command with thread-safe printing
    with print_lock:
        print(f"Executing: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Check for errors
    if result.returncode != 0:
        with print_lock:
            print(f"Error executing command: {result.stderr}")
        return params, None
    
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
        with print_lock:
            print("Could not extract MSE loss from output")
            print("Last 10 lines of output:")
            print(last_lines)
        return params, None
    
    return params, mse_loss

def grid_search(max_workers=4):
    # Define results directory
    results_dir = '../grid_search_results'
    os.makedirs(results_dir, exist_ok=True)
    
    # Define hyperparameter values to explore
    param_grid = {
        'layer': [2],
        'batch-size': [64],
        'lr': [0.0001, 0.0002],
        'tv-weight': [0.025, 0.05],
        'patience': [2],
        'epochs': [14],  # Fixed epochs since we have early stopping
        'no-cuda': [True],  # Use CPU for testing
        'save-model': [True]
    }
    
    # Create all possible parameter combinations
    keys = list(param_grid.keys())
    combinations = list(itertools.product(*[param_grid[key] for key in keys]))
    
    # Prepare list to save results
    results = []
    
    # Current date for filenames
    current_date = datetime.now().strftime("%Y%m%d")
    
    # Execute grid search in parallel
    total_combinations = len(combinations)
    print(f"Starting grid search with {total_combinations} combinations using {max_workers} workers...")
    
    # Create a ThreadPoolExecutor to execute tasks in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks to the executor
        futures = []
        for i, values in enumerate(combinations):
            params = dict(zip(keys, values))
            with print_lock:
                print(f"\nSubmitting combination {i+1}/{total_combinations}:")
                print(params)
            futures.append(executor.submit(run_attack, params))
        
        # Process results as they complete
        for i, future in enumerate(futures):
            params, mse_loss = future.result()
            
            with print_lock:
                print(f"\nCompleted combination {i+1}/{total_combinations}:")
                print(f"Parameters: {params}")
                print(f"MSE Loss: {mse_loss}")
            
            # Save result
            result_params = params.copy()
            result_params['mse_loss'] = mse_loss
            results.append(result_params)
            
            # Save partial results after each execution
            with print_lock:
                df = pd.DataFrame(results)
                csv_path = os.path.join(results_dir, f'grid_search_results_{current_date}.csv')
                df.to_csv(csv_path, index=False)
                
                # Show best result so far
                if len(results) > 0 and any(pd.notna(df['mse_loss'])):
                    # Add check to avoid NaN errors
                    best_idx = df['mse_loss'].idxmin()
                    if pd.notna(best_idx):
                        print(f"\nBest result so far:")
                        print(f"MSE Loss: {df.loc[best_idx, 'mse_loss']}")
                        print(f"Parameters: {df.loc[best_idx].to_dict()}")
    
    # Final results analysis
    df = pd.DataFrame(results)
    
    # Find the best combination
    if any(pd.notna(df['mse_loss'])):
        best_idx = df['mse_loss'].idxmin()
        best_params = df.loc[best_idx]
        
        print("\n" + "="*50)
        print("GRID SEARCH RESULTS")
        print("="*50)
        print(f"Best MSE Loss: {best_params['mse_loss']}")
        print("Best hyperparameters:")
        for key in keys:
            print(f"  {key}: {best_params[key]}")
        
        # Save final results
        final_csv_path = os.path.join(results_dir, f'grid_search_final_results_{current_date}.csv')
        df.sort_values('mse_loss').to_csv(final_csv_path, index=False)
        
        return best_params
    else:
        print("\nNo valid results found. Check for errors in the attack script execution.")
        return None

if __name__ == "__main__":
    # Create results directories if they don't exist
    os.makedirs('../grid_search_results', exist_ok=True)
    
    # Execute parallel grid search (adjust the number of workers as needed)
    best_params = grid_search(max_workers=4)
    
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