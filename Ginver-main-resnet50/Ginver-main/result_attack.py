import os
import torch
import torchvision.transforms as transforms
from Model import ResnetInversion_Generic, ResNet50EMBL
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import glob
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Set environment variable to avoid OpenMP error
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Path to the folder containing all layer subfolders with trained models
MODEL_ROOT = "../backups/backup03/ModelResult/blackbox"  # Change as needed

# Load classifier model (same for all layers)
model_path = "../ModelResult/classifier/classifier.pth"
model = torch.load(model_path, map_location=device)
classifier = ResNet50EMBL(model).to(device)
classifier.eval()
print("Classifier model loaded.")

# Prepare transform for images - use SAME transform as in attack.py
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),  # ResNet50 expects 224x224 images
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet normalization
        std=[0.229, 0.224, 0.225]
    )
])

# Function to load images from attack_test directory
def load_test_images(test_dir):
    image_paths = glob.glob(os.path.join(test_dir, "*.jpg")) + glob.glob(os.path.join(test_dir, "*.png"))
    if not image_paths:
        print(f"No images found in {test_dir}")
        exit(1)
    
    print(f"Found {len(image_paths)} test images")
    return image_paths

# Function to preprocess individual images
def preprocess_image(image_path):
    # Open image and convert to RGB (3 channels) as expected by ResNet50
    image = Image.open(image_path)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    return transform(image).unsqueeze(0)  # Add batch dimension

def denormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(tensor.device)
    return tensor * std + mean

# Path to test images
test_dir = "attack_test"
test_image_paths = load_test_images(os.path.join(".", test_dir))
num_images = min(5, len(test_image_paths))

# Layer configs for nz (output channels) - must match your training!
layer_nz_dict = {
    'conv1': 64,
    'relu1': 64,
    'maxpool': 64,
    'layer1': 256,
    'layer1_0': 256, 'layer1_1': 256, 'layer1_2': 256,
    'layer2': 512,
    'layer2_0': 512, 'layer2_1': 512, 'layer2_2': 512, 'layer2_3': 512,
    'layer3': 1024,
    'layer3_0': 1024, 'layer3_1': 1024, 'layer3_2': 1024, 'layer3_3': 1024, 'layer3_4': 1024, 'layer3_5': 1024,
    'layer4': 2048,
    'layer4_0': 2048, 'layer4_1': 2048, 'layer4_2': 2048,
}

RESULTS_DIR = "../results_visualization"
os.makedirs(RESULTS_DIR, exist_ok=True)

print("Folders found in MODEL_ROOT:", os.listdir(MODEL_ROOT))

for layer_name in sorted(os.listdir(MODEL_ROOT)):
    layer_path = os.path.join(MODEL_ROOT, layer_name)
    print(f"Checking {layer_path} ... is dir? {os.path.isdir(layer_path)}")
    if not os.path.isdir(layer_path):
        continue

    print(f"\nProcessing layer: {layer_name}")

    # Try to load the inversion model for this layer
    inversion = None
    nz = layer_nz_dict.get(layer_name, 1024)
    try:
        inversion = ResnetInversion_Generic(nc=3, ngf=64, nz=nz).to(device)
        checkpoint_path = os.path.join(layer_path, "inversion.pth")
        if not os.path.exists(checkpoint_path):
            checkpoint_path = os.path.join(layer_path, "final_inversion.pth")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        inversion.load_state_dict(checkpoint['model'])
        inversion.eval()
        print(f"Inversion model loaded from {checkpoint_path}")
    except Exception as e:
        print(f"Could not load inversion model for layer {layer_name}: {e}")
        continue

    fig, axes = plt.subplots(num_images, 3, figsize=(15, 15))
    mse_values = []
    ssim_values = []

    for i, image_path in enumerate(test_image_paths[:num_images]):
        input_tensor = preprocess_image(image_path).to(device)
        with torch.no_grad():
            activation = classifier(input_tensor, layer_name=layer_name)
            inverted_image = inversion(activation)
        original_img_denorm = denormalize(input_tensor)
        inverted_img_denorm = torch.clamp(inverted_image, 0, 1)
        original_img_np = original_img_denorm.cpu().numpy().squeeze().transpose(1, 2, 0)
        inverted_img_np = inverted_img_denorm.cpu().numpy().squeeze().transpose(1, 2, 0)
        original_gray = np.mean(original_img_np, axis=2) if original_img_np.shape[-1] == 3 else original_img_np
        inverted_gray = np.mean(inverted_img_np, axis=2) if inverted_img_np.shape[-1] == 3 else inverted_img_np
        mse_value = np.mean((original_gray - inverted_gray) ** 2)
        ssim_value = ssim(original_gray, inverted_gray, data_range=original_gray.max() - original_gray.min())
        mse_values.append(mse_value)
        ssim_values.append(ssim_value)
        axes[i, 0].imshow(original_gray, cmap='gray')
        axes[i, 0].set_title(f'Original Image {i+1} (Grayscale)\n{os.path.basename(image_path)}')
        axes[i, 0].axis('off')
        axes[i, 1].imshow(inverted_gray, cmap='gray')
        axes[i, 1].set_title(f'Reconstructed Image {i+1} (Grayscale)')
        axes[i, 1].axis('off')
        axes[i, 2].text(0.5, 0.5, f'MSE: {mse_value:.4f}\nSSIM: {ssim_value:.4f}',
                       horizontalalignment='center', verticalalignment='center', fontsize=12)
        axes[i, 2].axis('off')

    plt.suptitle(f'Layer: {layer_name} - Average MSE: {np.mean(mse_values):.4f}, Average SSIM: {np.mean(ssim_values):.4f}')
    plt.tight_layout()
    result_path = os.path.join(RESULTS_DIR, f'blackbox_{layer_name}_attack_test.png')
    plt.savefig(result_path)
    print(f"Results saved to {result_path}")