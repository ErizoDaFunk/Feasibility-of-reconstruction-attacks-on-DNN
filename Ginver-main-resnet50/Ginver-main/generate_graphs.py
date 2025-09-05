import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Create output directory
output_dir = 'metrics_graphs'
os.makedirs(output_dir, exist_ok=True)

# Read the metrics CSV file
df = pd.read_csv('metrics.csv')

# Filter only blackbox mode data
blackbox_data = df[df['mode'] == 'blackbox'].copy()

# Get layer names in order and metrics
layers = blackbox_data['layer'].tolist()
mse_values = blackbox_data['mse'].tolist()
ssim_values = blackbox_data['ssim'].tolist()
lpips_values = blackbox_data['lpips'].tolist()
class_acc_values = blackbox_data['class_acc'].tolist()

# Create x-axis positions
x_pos = np.arange(len(layers))

# Set up the plotting style
plt.style.use('default')
fig_width, fig_height = 12, 6

# 1. MSE Evolution Graph
plt.figure(figsize=(fig_width, fig_height))
plt.plot(x_pos, mse_values, marker='o', linewidth=2, markersize=6, color='red', label='MSE')
plt.xlabel('Layer', fontsize=12)
plt.ylabel('MSE Value', fontsize=12)
plt.title('MSE Evolution Across ResNet50 Layers', fontsize=14, fontweight='bold')
plt.xticks(x_pos, layers, rotation=45, ha='right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'mse_evolution.png'), dpi=300, bbox_inches='tight')
plt.close()

# 2. SSIM Evolution Graph
plt.figure(figsize=(fig_width, fig_height))
plt.plot(x_pos, ssim_values, marker='s', linewidth=2, markersize=6, color='blue', label='SSIM')
plt.xlabel('Layer', fontsize=12)
plt.ylabel('SSIM Value', fontsize=12)
plt.title('SSIM Evolution Across ResNet50 Layers', fontsize=14, fontweight='bold')
plt.xticks(x_pos, layers, rotation=45, ha='right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'ssim_evolution.png'), dpi=300, bbox_inches='tight')
plt.close()

# 3. LPIPS Evolution Graph
plt.figure(figsize=(fig_width, fig_height))
plt.plot(x_pos, lpips_values, marker='^', linewidth=2, markersize=6, color='green', label='LPIPS')
plt.xlabel('Layer', fontsize=12)
plt.ylabel('LPIPS Value', fontsize=12)
plt.title('LPIPS Evolution Across ResNet50 Layers', fontsize=14, fontweight='bold')
plt.xticks(x_pos, layers, rotation=45, ha='right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'lpips_evolution.png'), dpi=300, bbox_inches='tight')
plt.close()

# 4. Classification Accuracy Evolution Graph
plt.figure(figsize=(fig_width, fig_height))
plt.plot(x_pos, class_acc_values, marker='D', linewidth=2, markersize=6, color='purple', label='Classification Accuracy')
plt.xlabel('Layer', fontsize=12)
plt.ylabel('Classification Accuracy', fontsize=12)
plt.title('Classification Accuracy Evolution Across ResNet50 Layers', fontsize=14, fontweight='bold')
plt.xticks(x_pos, layers, rotation=45, ha='right')
plt.grid(True, alpha=0.3)
plt.ylim(0, 1)  # Accuracy is typically between 0 and 1
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'accuracy_evolution.png'), dpi=300, bbox_inches='tight')
plt.close()

# 5. Combined graph with reconstruction metrics only (normalized) - ACCURACY REMOVED
plt.figure(figsize=(14, 8))

# Normalize values to [0, 1] for comparison - only reconstruction metrics
mse_norm = (np.array(mse_values) - min(mse_values)) / (max(mse_values) - min(mse_values))
ssim_norm = (np.array(ssim_values) - min(ssim_values)) / (max(ssim_values) - min(ssim_values))
lpips_norm = (np.array(lpips_values) - min(lpips_values)) / (max(lpips_values) - min(lpips_values))

plt.plot(x_pos, mse_norm, marker='o', linewidth=2, markersize=6, color='red', label='MSE (normalized)')
plt.plot(x_pos, ssim_norm, marker='s', linewidth=2, markersize=6, color='blue', label='SSIM (normalized)')
plt.plot(x_pos, lpips_norm, marker='^', linewidth=2, markersize=6, color='green', label='LPIPS (normalized)')

plt.xlabel('Layer', fontsize=12)
plt.ylabel('Normalized Value', fontsize=12)
plt.title('Reconstruction Metrics Evolution Across ResNet50 Layers (Normalized)', fontsize=14, fontweight='bold')
plt.xticks(x_pos, layers, rotation=45, ha='right')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'combined_metrics_evolution.png'), dpi=300, bbox_inches='tight')
plt.close()

# 6. Classification performance metrics only
plt.figure(figsize=(fig_width, fig_height))
plt.plot(x_pos, class_acc_values, marker='D', linewidth=2, markersize=6, color='purple', label='Classification Accuracy')
plt.xlabel('Layer', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('Reconstruction Quality vs Classification Accuracy', fontsize=14, fontweight='bold')
plt.xticks(x_pos, layers, rotation=45, ha='right')
plt.grid(True, alpha=0.3)
plt.ylim(0, 1)
# Add percentage labels on points
for i, acc in enumerate(class_acc_values):
    plt.annotate(f'{acc:.2%}', xy=(i, acc), xytext=(5, 5), 
                textcoords='offset points', fontsize=9, ha='left')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'classification_accuracy_detailed.png'), dpi=300, bbox_inches='tight')
plt.close()

print("Graphs generated successfully!")
print(f"Files created in '{output_dir}' folder:")
print("- mse_evolution.png")
print("- ssim_evolution.png") 
print("- lpips_evolution.png")
print("- accuracy_evolution.png")
print("- classification_accuracy_detailed.png")
print("- combined_metrics_evolution.png (reconstruction metrics only)")

# Print some statistics
print(f"\nStatistics:")
print(f"MSE - Min: {min(mse_values):.6f}, Max: {max(mse_values):.6f}")
print(f"SSIM - Min: {min(ssim_values):.6f}, Max: {max(ssim_values):.6f}")
print(f"LPIPS - Min: {min(lpips_values):.6f}, Max: {max(lpips_values):.6f}")
print(f"Classification Accuracy - Min: {min(class_acc_values):.6f}, Max: {max(class_acc_values):.6f}")
print(f"Average Classification Accuracy: {np.mean(class_acc_values):.6f}")