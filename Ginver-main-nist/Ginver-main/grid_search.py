import subprocess
import os
import itertools
import pandas as pd
import re
from datetime import datetime

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

def grid_search():
    # Define hyperparameter values to explore
    param_grid = {
        'layer': [2],
        'batch-size': [64],
        'lr': [0.0001, 0.0002, 0.0005],
        'tv-weight': [0.025, 0.05, 0.1],
        'patience': [2],
        'epochs': [14],  # Fixed epochs since we have early stopping
        'no-cuda': [True],  # Use CPU for testing
        'save-model': [True]
    }
    
    # Create all possible parameter combinations
    keys = list(param_grid.keys())
    combinations = list(itertools.product(*[param_grid[key] for key in keys]))
    
    # Prepare DataFrame to save results
    results = []
    
    # Execute grid search
    total_combinations = len(combinations)
    for i, values in enumerate(combinations):
        params = dict(zip(keys, values))
        print(f"\nCombination {i+1}/{total_combinations}:")
        print(params)
        
        # Execute attack.py with current parameters
        mse_loss = run_attack(params)
        
        # Save results
        params['mse_loss'] = mse_loss
        results.append(params)
        
        # Save partial results after each execution
        df = pd.DataFrame(results)
        df.to_csv(f'grid_search_results_{datetime.now().strftime("%Y%m%d")}.csv', index=False)
        
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
        df.sort_values('mse_loss').to_csv(f'grid_search_final_results_{datetime.now().strftime("%Y%m%d")}.csv', index=False)
        
        return best_params
    else:
        print("\nNo valid results found. Check for errors in the attack script execution.")
        return None

if __name__ == "__main__":
    # Create results directories if they don't exist
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