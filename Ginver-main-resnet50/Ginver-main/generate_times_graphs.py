import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Create output directory
output_dir = 'metrics_graphs'
os.makedirs(output_dir, exist_ok=True)

# Read the grid search results CSV file
df = pd.read_csv('../grid_search_results/grid_search_final_results.csv')

# Filter only blackbox mode data (if there are other modes)
blackbox_data = df[df['mode'] == 'blackbox'].copy()

# Get layer names in order and training times
layers = blackbox_data['layer'].tolist()
training_times = blackbox_data['training_time'].tolist()

# Convert training times from seconds to minutes for better readability
training_times_minutes = [time/60 for time in training_times]

# Create x-axis positions
x_pos = np.arange(len(layers))

# Set up the plotting style
plt.style.use('default')
fig_width, fig_height = 14, 8

# 1. Training Time Evolution Graph (in minutes)
plt.figure(figsize=(fig_width, fig_height))
plt.plot(x_pos, training_times_minutes, marker='o', linewidth=2, markersize=8, color='green', label='Training Time')
plt.xlabel('Layer', fontsize=12)
plt.ylabel('Training Time (minutes)', fontsize=12)
plt.title('Training Time Evolution Across ResNet50 Layers', fontsize=14, fontweight='bold')
plt.xticks(x_pos, layers, rotation=45, ha='right')
plt.grid(True, alpha=0.3)

# Add value labels on each point
for i, (layer, time_min) in enumerate(zip(layers, training_times_minutes)):
    plt.annotate(f'{time_min:.1f}m', 
                xy=(i, time_min), 
                xytext=(5, 5), 
                textcoords='offset points',
                fontsize=9,
                ha='left')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'training_times_evolution.png'), dpi=300, bbox_inches='tight')
plt.close()

# 2. Training Time Evolution Graph (in hours for longer times)
plt.figure(figsize=(fig_width, fig_height))
training_times_hours = [time/3600 for time in training_times]
plt.plot(x_pos, training_times_hours, marker='s', linewidth=2, markersize=8, color='orange', label='Training Time')
plt.xlabel('Layer', fontsize=12)
plt.ylabel('Training Time (hours)', fontsize=12)
plt.title('Training Time Evolution Across ResNet50 Layers', fontsize=14, fontweight='bold')
plt.xticks(x_pos, layers, rotation=45, ha='right')
plt.grid(True, alpha=0.3)

# Add value labels on each point
for i, (layer, time_hour) in enumerate(zip(layers, training_times_hours)):
    plt.annotate(f'{time_hour:.2f}h', 
                xy=(i, time_hour), 
                xytext=(5, 5), 
                textcoords='offset points',
                fontsize=9,
                ha='left')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'training_times_evolution_hours.png'), dpi=300, bbox_inches='tight')
plt.close()

# 3. Bar chart for better comparison
plt.figure(figsize=(fig_width, fig_height))
bars = plt.bar(x_pos, training_times_minutes, color='skyblue', alpha=0.7, edgecolor='navy')
plt.xlabel('Layer', fontsize=12)
plt.ylabel('Training Time (minutes)', fontsize=12)
plt.title('Training Time Comparison Across ResNet50 Layers', fontsize=14, fontweight='bold')
plt.xticks(x_pos, layers, rotation=45, ha='right')
plt.grid(True, alpha=0.3, axis='y')

# Add value labels on top of bars
for i, (bar, time_min) in enumerate(zip(bars, training_times_minutes)):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(training_times_minutes)*0.01,
             f'{time_min:.1f}m', 
             ha='center', va='bottom', fontsize=9, rotation=90)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'training_times_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()

# 4. Log scale plot for better visualization of the range
plt.figure(figsize=(fig_width, fig_height))
plt.plot(x_pos, training_times, marker='D', linewidth=2, markersize=8, color='red', label='Training Time (log scale)')
plt.xlabel('Layer', fontsize=12)
plt.ylabel('Training Time (seconds, log scale)', fontsize=12)
plt.title('Training Time Evolution Across ResNet50 Layers (Log Scale)', fontsize=14, fontweight='bold')
plt.yscale('log')
plt.xticks(x_pos, layers, rotation=45, ha='right')
plt.grid(True, alpha=0.2, which="both", ls="-")  # Fixed: removed duplicate alpha

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'training_times_log_scale.png'), dpi=300, bbox_inches='tight')
plt.close()

# Print summary statistics
print("Training Time Statistics:")
print("=" * 50)
print(f"Fastest training: {min(training_times_minutes):.1f} minutes ({layers[training_times_minutes.index(min(training_times_minutes))]})")
print(f"Slowest training: {max(training_times_minutes):.1f} minutes ({layers[training_times_minutes.index(max(training_times_minutes))]})")
print(f"Average training time: {np.mean(training_times_minutes):.1f} minutes")
print(f"Total training time: {sum(training_times_hours):.2f} hours")

# Create a summary table
summary_df = pd.DataFrame({
    'Layer': layers,
    'Training_Time_Minutes': [f"{t:.1f}" for t in training_times_minutes],
    'Training_Time_Hours': [f"{t:.2f}" for t in training_times_hours]
})

summary_df.to_csv(os.path.join(output_dir, 'training_times_summary.csv'), index=False)

print(f"\nGraphs saved in '{output_dir}' directory:")
print("- training_times_evolution.png (minutes)")
print("- training_times_evolution_hours.png (hours)")
print("- training_times_comparison.png (bar chart)")
print("- training_times_log_scale.png (log scale)")
print("- training_times_summary.csv (summary table)")