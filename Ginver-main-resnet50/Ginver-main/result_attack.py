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
import lpips

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Set environment variable to avoid OpenMP error
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize LPIPS model
lpips_model = lpips.LPIPS(net='alex').to(device)  # or 'vgg', 'squeeze'

# Path to the folder containing all layer subfolders with trained models
MODEL_ROOT = "../backups/backup03/ModelResult/blackbox"  # Change as needed

# Load classifier model (same for all layers)
model_path = "../ModelResult/classifier/classifier.pth"
model = torch.load(model_path, map_location=device, weights_only=False)
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

# Function to convert display layer name to actual layer name for classifier
def get_actual_layer_name(display_name):
    """Convert display name (block1, block2, etc.) to actual layer name for classifier"""
    layer_mapping = {
        'block1': 'layer1',
        'block2': 'layer2', 
        'block3': 'layer3',
        'block4': 'layer4'
    }
    return layer_mapping.get(display_name, display_name)

# Function to get display layer name from actual layer name
def get_display_layer_name(actual_name):
    """Convert actual layer name to display name (layer1 -> block1, etc.)"""
    display_mapping = {
        'layer1': 'block1',
        'layer2': 'block2',
        'layer3': 'block3', 
        'layer4': 'block4'
    }
    return display_mapping.get(actual_name, actual_name)

# Path to test images
test_dir = "attack_test"
test_image_paths = load_test_images(os.path.join(".", test_dir))
num_images = min(5, len(test_image_paths))

# Layer configs for nz (output channels) - must match your training!
# Using actual layer names for the dictionary keys
layer_nz_dict = {
    'conv1': 64,
    'relu1': 64,
    'maxpool': 64,
    'layer1': 256,  # This will be displayed as block1
    'layer1_0': 256, 'layer1_1': 256, 'layer1_2': 256,
    'layer2': 512,  # This will be displayed as block2
    'layer2_0': 512, 'layer2_1': 512, 'layer2_2': 512, 'layer2_3': 512,
    'layer3': 1024,  # This will be displayed as block3
    'layer3_0': 1024, 'layer3_1': 1024, 'layer3_2': 1024, 'layer3_3': 1024, 'layer3_4': 1024, 'layer3_5': 1024,
    'layer4': 2048,  # This will be displayed as block4
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
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        inversion.load_state_dict(checkpoint['model'])
        inversion.eval()
        print(f"Inversion model loaded from {checkpoint_path}")
    except Exception as e:
        print(f"Could not load inversion model for layer {layer_name}: {e}")
        continue

    # Create figure with minimal spacing between image columns
    fig = plt.figure(figsize=(14, 15))
    gs = fig.add_gridspec(num_images, 3, width_ratios=[1, 1, 1], wspace=0.005, hspace=0.2)
    
    mse_values = []
    ssim_values = []
    lpips_values = []

    for i, image_path in enumerate(test_image_paths[:num_images]):
        input_tensor = preprocess_image(image_path).to(device)
        
        with torch.no_grad():
            # Use actual layer name for classifier (convert if necessary)
            actual_layer_name = get_actual_layer_name(layer_name)
            activation = classifier(input_tensor, layer_name=actual_layer_name)
            inverted_image = inversion(activation)
        
        # Denormalize for display and metrics
        original_img_denorm = denormalize(input_tensor).clamp(0, 1)
        inverted_img_denorm = inverted_image.clamp(0, 1)
        
        # Convert to numpy for display and MSE/SSIM computation
        original_img_np = original_img_denorm.cpu().numpy().squeeze().transpose(1, 2, 0)
        inverted_img_np = inverted_img_denorm.cpu().numpy().squeeze().transpose(1, 2, 0)
        
        # Convert to grayscale for MSE and SSIM
        original_gray = np.mean(original_img_np, axis=2) if original_img_np.shape[-1] == 3 else original_img_np
        inverted_gray = np.mean(inverted_img_np, axis=2) if inverted_img_np.shape[-1] == 3 else inverted_img_np
        
        # Compute MSE and SSIM
        mse_value = np.mean((original_gray - inverted_gray) ** 2)
        ssim_value = ssim(original_gray, inverted_gray, data_range=original_gray.max() - original_gray.min())
        
        # Compute LPIPS (requires tensors in [-1, 1] range)
        with torch.no_grad():
            # Convert from [0, 1] to [-1, 1] for LPIPS
            orig_lpips = original_img_denorm * 2.0 - 1.0
            recon_lpips = inverted_img_denorm * 2.0 - 1.0
            lpips_value = lpips_model(orig_lpips, recon_lpips).item()
        
        mse_values.append(mse_value)
        ssim_values.append(ssim_value)
        lpips_values.append(lpips_value)
        
        # Create subplots using gridspec
        ax1 = fig.add_subplot(gs[i, 0])
        ax2 = fig.add_subplot(gs[i, 1])
        ax3 = fig.add_subplot(gs[i, 2])
        
        # Original image
        ax1.imshow(original_gray, cmap='gray')
        ax1.set_title(f'Original Image {i+1}\n{os.path.basename(image_path)}', fontsize=9)
        ax1.axis('off')
        
        # Reconstructed image
        ax2.imshow(inverted_gray, cmap='gray')
        ax2.set_title(f'Reconstructed Image {i+1}', fontsize=9)
        ax2.axis('off')
        
        # Metrics column (positioned right next to reconstructed images)
        metrics_text = f'MSE: {mse_value:.4f}\nSSIM: {ssim_value:.4f}\nLPIPS: {lpips_value:.4f}'
        ax3.text(0.02, 0.5, metrics_text, horizontalalignment='left', verticalalignment='center', 
                fontsize=10, transform=ax3.transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        ax3.axis('off')

    # Overall title with average metrics - use display name
    display_layer_name = get_display_layer_name(layer_name)
    avg_mse = np.mean(mse_values)
    avg_ssim = np.mean(ssim_values)
    avg_lpips = np.mean(lpips_values)
    
    plt.suptitle(f'Layer: {display_layer_name} - Avg MSE: {avg_mse:.4f}, Avg SSIM: {avg_ssim:.4f}, Avg LPIPS: {avg_lpips:.4f}', 
                fontsize=14, y=0.98)
    
    # Use display name for filename
    result_path = os.path.join(RESULTS_DIR, f'blackbox_{display_layer_name}_attack_test.png')
    plt.savefig(result_path, dpi=150, bbox_inches='tight')
    plt.close()  # Close figure to free memory
    
    print(f"Results saved to {result_path}")
    print(f"  Average MSE: {avg_mse:.4f}, Average SSIM: {avg_ssim:.4f}, Average LPIPS: {avg_lpips:.4f}")