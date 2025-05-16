import os
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision import datasets
from Model import ResNetInversion_Conv1, ResNet50EMBL
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

# Define the layer to analyze
layer_name = "conv1"  # Change this to the layer you want to analyze
mode = "whitebox"  # Same as in attack.py

# Load classifier model
model_path = "../ModelResult/classifier/classifier.pth"
model = torch.load(model_path, map_location=device)
classifier = ResNet50EMBL(model).to(device)
classifier.eval()
print("Classifier model loaded.")

# Create inversion model with the SAME class used in attack.py
inversion = ResNetInversion_Conv1(nc=3).to(device)  # Changed to match attack.py

# Try to load the saved inversion model
try:
    checkpoint_path = f'../ModelResult/{mode}/{layer_name}/inversion.pth'  # Match path in attack.py
    print(f"Attempting to load model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    inversion.load_state_dict(checkpoint['model'])
    print(f"Inversion model loaded from {checkpoint_path}")
    print(f"Epoch: {checkpoint['epoch']}, Best MSE: {checkpoint['best_mse_loss']:.5f}")
except Exception as e:
    print(f"Error loading inversion model: {e}")
    # Try the final inversion path as fallback
    try:
        final_path = f'../ModelResult/{mode}/{layer_name}/final_inversion.pth'
        print(f"Attempting to load model from {final_path}")
        checkpoint = torch.load(final_path, map_location=device)
        inversion.load_state_dict(checkpoint['model'])
        print(f"Inversion model loaded from {final_path}")
        print(f"Epoch: {checkpoint['epoch']}, Best MSE: {checkpoint['best_mse_loss']:.5f}")
    except Exception as e:
        print(f"Error loading final inversion model: {e}")
        exit(1)

inversion.eval()

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

# Load images from attack_test directory
test_dir = "attack_test"
test_image_paths = load_test_images(os.path.join(".", test_dir))

# Function to denormalize images for visualization
def denormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(tensor.device)
    return tensor * std + mean

# Generate and display inverted images
num_images = min(5, len(test_image_paths))
fig, axes = plt.subplots(num_images, 3, figsize=(15, 15))

mse_values = []
ssim_values = []

for i, image_path in enumerate(test_image_paths[:num_images]):
    # Load and preprocess image
    input_tensor = preprocess_image(image_path)
    input_tensor = input_tensor.to(device)
    
    # Get classifier output for the specific layer
    with torch.no_grad():
        activation = classifier(input_tensor, layer_name=layer_name)
    
    # Generate inverted image
    with torch.no_grad():
        inverted_image = inversion(activation)
    
    # Denormalize for visualization
    original_img_denorm = denormalize(input_tensor)
    inverted_img_denorm = torch.clamp(inverted_image, 0, 1)  # Ensure values are in [0,1]
    
    # Convert to numpy for metrics and display
    original_img_np = original_img_denorm.cpu().numpy().squeeze().transpose(1, 2, 0)  # Change from CxHxW to HxWxC
    inverted_img_np = inverted_img_denorm.cpu().numpy().squeeze().transpose(1, 2, 0)
    
    # Calculate metrics (convert to grayscale for SSIM if using RGB)
    original_gray = np.mean(original_img_np, axis=2) if original_img_np.shape[-1] == 3 else original_img_np
    inverted_gray = np.mean(inverted_img_np, axis=2) if inverted_img_np.shape[-1] == 3 else inverted_img_np
    
    mse_value = np.mean((original_gray - inverted_gray) ** 2)
    ssim_value = ssim(original_gray, inverted_gray, data_range=original_gray.max() - original_gray.min())
    
    mse_values.append(mse_value)
    ssim_values.append(ssim_value)
    
    # Display original image
    axes[i, 0].imshow(original_img_np)
    axes[i, 0].set_title(f'Original Image {i+1}\n{os.path.basename(image_path)}')
    axes[i, 0].axis('off')
    
    # Display inverted image
    axes[i, 1].imshow(inverted_img_np)
    axes[i, 1].set_title(f'Reconstructed Image {i+1}')
    axes[i, 1].axis('off')
    
    # Display metrics
    axes[i, 2].text(0.5, 0.5, f'MSE: {mse_value:.4f}\nSSIM: {ssim_value:.4f}', 
                   horizontalalignment='center', verticalalignment='center', fontsize=12)
    axes[i, 2].axis('off')

# Add overall metrics summary
plt.suptitle(f'Layer: {layer_name} - Average MSE: {np.mean(mse_values):.4f}, Average SSIM: {np.mean(ssim_values):.4f}')

plt.tight_layout()
# Save the figure
result_path = f'../results_visualization_{mode}_{layer_name}_attack_test.png'
plt.savefig(result_path)
print(f"Results saved to {result_path}")

# Show plot
plt.show()