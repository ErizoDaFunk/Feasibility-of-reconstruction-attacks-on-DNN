import os
import shutil
import random
from pathlib import Path

# Source directory with original EMBL dataset
source_dir = Path("../data/GS_original")  # Replace with actual path

# Target directories
data_dir = Path("../data/GS_organized")
train_dir = data_dir / "train"
test_dir = data_dir / "test"

# Create directory structure
os.makedirs(train_dir / "onsample", exist_ok=True)
os.makedirs(train_dir / "offsample", exist_ok=True)
os.makedirs(test_dir / "onsample", exist_ok=True)
os.makedirs(test_dir / "offsample", exist_ok=True)

# Find all onsample and offsample images recursively
onsample_images = []
offsample_images = []

for root, dirs, files in os.walk(source_dir):
    for file in files:
        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            filepath = os.path.join(root, file)
            if 'on' in str(filepath).lower():
                onsample_images.append(filepath)
            elif 'off' in str(filepath).lower():
                offsample_images.append(filepath)

# Shuffle images for randomization
random.seed(42)
random.shuffle(onsample_images)
random.shuffle(offsample_images)

# Split into train/test sets (80/20 split)
test_onsample_size = int(len(onsample_images) * 0.2)
test_offsample_size = int(len(offsample_images) * 0.2)

train_onsample = onsample_images[test_onsample_size:]
test_onsample = onsample_images[:test_onsample_size]

train_offsample = offsample_images[test_offsample_size:]
test_offsample = offsample_images[:test_offsample_size]

# Copy files to their respective directories
def copy_files(file_list, dest_dir):
    for i, file_path in enumerate(file_list):
        dest_file = dest_dir / f"img_{i:05d}{Path(file_path).suffix}"
        shutil.copy2(file_path, dest_file)
    return len(file_list)

# Copy files
print(f"Copying {len(train_onsample)} onsample images to training set...")
copy_files(train_onsample, train_dir / "onsample")

print(f"Copying {len(train_offsample)} offsample images to training set...")
copy_files(train_offsample, train_dir / "offsample")

print(f"Copying {len(test_onsample)} onsample images to test set...")
copy_files(test_onsample, test_dir / "onsample")

print(f"Copying {len(test_offsample)} offsample images to test set...")
copy_files(test_offsample, test_dir / "offsample")

print("Dataset organization complete!")

# Print summary
print("\nDataset Summary:")
print(f"Total onsample images: {len(onsample_images)}")
print(f"Total offsample images: {len(offsample_images)}")
print(f"Training onsample: {len(train_onsample)}")
print(f"Training offsample: {len(train_offsample)}")
print(f"Testing onsample: {len(test_onsample)}")
print(f"Testing offsample: {len(test_offsample)}")