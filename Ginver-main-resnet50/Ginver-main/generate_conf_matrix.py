import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Path to metrics.csv
METRICS_CSV = "metrics.csv"
OUTPUT_DIR = "confusion_matrices"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Read the metrics.csv file
df = pd.read_csv(METRICS_CSV)

for idx, row in df.iterrows():
    layer = row['layer']
    tp = int(row['true_positives'])
    tn = int(row['true_negatives'])
    fp = int(row['false_positives'])
    fn = int(row['false_negatives'])

    # Confusion matrix: rows = true (onsample, offsample), cols = predicted (onsample, offsample)
    # [[TP, FN], [FP, TN]]
    # But for binary classification, sklearn and seaborn expect:
    # [[TN, FP], [FN, TP]] with labels [offsample, onsample]
    cm = np.array([[tn, fp],
                   [fn, tp]])

    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['offsample', 'onsample'],
                yticklabels=['offsample', 'onsample'])
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title(f'Confusion Matrix - {layer}')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'conf_matrix_{layer}.png'))
    plt.close()

print(f"Confusion matrices saved in {OUTPUT_DIR}")