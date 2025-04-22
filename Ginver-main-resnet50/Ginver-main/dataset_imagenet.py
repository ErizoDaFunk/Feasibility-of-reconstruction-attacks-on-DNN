import os
import urllib.request
import zipfile
import shutil
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Ruta base del proyecto
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
ZIP_FILE = os.path.join(DATA_DIR, 'tiny-imagenet-200.zip')
EXTRACTED_DIR = os.path.join(DATA_DIR, 'tiny-imagenet-200')
TINY_IMAGENET_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"

def download_and_extract():
    os.makedirs(DATA_DIR, exist_ok=True)

    if not os.path.exists(ZIP_FILE):
        print("📥 Downloading Tiny ImageNet...")
        urllib.request.urlretrieve(TINY_IMAGENET_URL, ZIP_FILE)
        print("✅ Download complete.")
    else:
        print("✅ Zip file already exists.")

    if not os.path.exists(EXTRACTED_DIR):
        print("📦 Extracting...")
        with zipfile.ZipFile(ZIP_FILE, 'r') as zip_ref:
            zip_ref.extractall(DATA_DIR)
        print("✅ Extraction complete.")
    else:
        print("✅ Dataset already extracted.")

def process_val_folder():
    val_dir = os.path.join(EXTRACTED_DIR, 'val')
    images_dir = os.path.join(val_dir, 'images')
    annotations_file = os.path.join(val_dir, 'val_annotations.txt')

    print("🔧 Organizing validation set...")
    with open(annotations_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            img_file, class_id = parts[0], parts[1]
            class_dir = os.path.join(val_dir, class_id)
            os.makedirs(class_dir, exist_ok=True)

            src_path = os.path.join(images_dir, img_file)
            dst_path = os.path.join(class_dir, img_file)

            if os.path.exists(src_path):
                shutil.move(src_path, dst_path)

    if os.path.exists(images_dir):
        shutil.rmtree(images_dir)

    print("✅ Validation set ready.")

def get_dataloaders(batch_size=64, num_workers=2):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    train_path = os.path.join(EXTRACTED_DIR, 'train')
    val_path = os.path.join(EXTRACTED_DIR, 'val')

    train_dataset = datasets.ImageFolder(train_path, transform=transform)
    val_dataset = datasets.ImageFolder(val_path, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader

if __name__ == "__main__":
    download_and_extract()
    process_val_folder()

    # Prueba rápida
    train_loader, val_loader = get_dataloaders()
    print(f"🔍 Train batches: {len(train_loader)} | Validation batches: {len(val_loader)}")
