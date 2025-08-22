import os
import torch
import torchvision.transforms as transforms
from Model import ResnetInversion_Generic, ResNet50EMBL
import numpy as np
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import pandas as pd
from glob import glob
import lpips

# Paths
TEST_ROOT = "../data/GS_organized/test"
MODEL_ROOT = "../backups/backup03/ModelResult/blackbox"
CLASSIFIER_PATH = "../ModelResult/classifier/classifier.pth"
METRICS_CSV = "metrics.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize LPIPS model
lpips_model = lpips.LPIPS(net='alex').to(device)  # or 'vgg', 'squeeze'

# ImageNet normalization (as used in training)
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def denormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(tensor.device)
    return tensor * std + mean

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

# Load classifier
model_weights = torch.load(CLASSIFIER_PATH, map_location=device, weights_only=False)
classifier = ResNet50EMBL(model_weights).to(device)
classifier.eval()

# Prepare test image paths and labels
def get_test_images():
    image_label_list = []
    for label_name in ['offsample', 'onsample']:
        folder = os.path.join(TEST_ROOT, label_name)
        for ext in ('*.jpg', '*.png', '*.jpeg', '*.bmp'):
            for img_path in glob(os.path.join(folder, ext)):
                image_label_list.append((img_path, label_name))
    return image_label_list

test_images = get_test_images()

# Ensure we have test images
if not test_images:
    raise ValueError("No test images found in the specified directory.")    

# print number of test images
print(f"Found {len(test_images)} test images.")

results = []

print(f"Looking for models in: {MODEL_ROOT}")
print(f"Available directories: {sorted(os.listdir(MODEL_ROOT))}")

for layer_name in sorted(os.listdir(MODEL_ROOT)):
    layer_path = os.path.join(MODEL_ROOT, layer_name)
    if not os.path.isdir(layer_path):
        print(f"Skipping {layer_name} - not a directory")
        continue

    print(f"\nProcessing layer: {layer_name}")
    
    # Load inversion model for this layer
    nz = layer_nz_dict.get(layer_name, 1024)
    print(f"Using nz={nz} for layer {layer_name}")
    
    try:
        inversion = ResnetInversion_Generic(nc=3, ngf=64, nz=nz).to(device)
        checkpoint_path = os.path.join(layer_path, "inversion.pth")
        if not os.path.exists(checkpoint_path):
            checkpoint_path = os.path.join(layer_path, "final_inversion.pth")
        
        print(f"Looking for checkpoint at: {checkpoint_path}")
        if not os.path.exists(checkpoint_path):
            print(f"No checkpoint found for layer {layer_name}")
            continue
            
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        inversion.load_state_dict(checkpoint['model'])
        inversion.eval()
        print(f"Successfully loaded model for layer {layer_name}")
    except Exception as e:
        print(f"Could not load inversion model for layer {layer_name}: {e}")
        continue

    mse_list = []
    ssim_list = []
    lpips_list = []
    
    # Confusion matrix counters
    true_positives = 0   # Predicted onsample, actually onsample
    true_negatives = 0   # Predicted offsample, actually offsample  
    false_positives = 0  # Predicted onsample, actually offsample
    false_negatives = 0  # Predicted offsample, actually onsample
    total = 0

    for img_path, label_name in test_images:
        # Prepare input
        image = Image.open(img_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(device)

        # Get activation and reconstruct
        with torch.no_grad():
            activation = classifier(input_tensor, layer_name=layer_name)
            reconstructed = inversion(activation)

        # Denormalize for metric computation
        original_denorm = denormalize(input_tensor).clamp(0, 1)
        reconstructed_denorm = reconstructed.clamp(0, 1)

        # Convert to numpy for MSE and SSIM
        orig_np = original_denorm.cpu().numpy().squeeze().transpose(1, 2, 0)
        recon_np = reconstructed_denorm.cpu().numpy().squeeze().transpose(1, 2, 0)

        # Convert to grayscale for MSE and SSIM
        orig_gray = np.mean(orig_np, axis=2) if orig_np.ndim == 3 else orig_np
        recon_gray = np.mean(recon_np, axis=2) if recon_np.ndim == 3 else recon_np

        # Compute MSE and SSIM
        mse_val = np.mean((orig_gray - recon_gray) ** 2)
        ssim_val = ssim(orig_gray, recon_gray, data_range=orig_gray.max() - orig_gray.min())
        mse_list.append(mse_val)
        ssim_list.append(ssim_val)

        # Compute LPIPS (requires tensors in [-1, 1] range)
        with torch.no_grad():
            # Convert from [0, 1] to [-1, 1] for LPIPS
            orig_lpips = original_denorm * 2.0 - 1.0
            recon_lpips = reconstructed_denorm * 2.0 - 1.0
            lpips_val = lpips_model(orig_lpips, recon_lpips).item()
            lpips_list.append(lpips_val)

        # Classification accuracy and confusion matrix
        with torch.no_grad():
            pred_logits = classifier(reconstructed_denorm, layer_name=None)
            pred_class = torch.argmax(pred_logits, dim=1).item()
            # 0: offsample, 1: onsample
            true_class = 0 if label_name == 'offsample' else 1
            
            # Update confusion matrix
            if true_class == 1 and pred_class == 1:
                true_positives += 1
            elif true_class == 0 and pred_class == 0:
                true_negatives += 1
            elif true_class == 0 and pred_class == 1:
                false_positives += 1
            elif true_class == 1 and pred_class == 0:
                false_negatives += 1
            
            total += 1

    avg_mse = float(np.mean(mse_list))
    avg_ssim = float(np.mean(ssim_list))
    avg_lpips = float(np.mean(lpips_list))
    acc = (true_positives + true_negatives) / total if total > 0 else 0.0

    results.append({
        "layer": layer_name,
        "mode": "blackbox",
        "mse": avg_mse,
        "ssim": avg_ssim,
        "lpips": avg_lpips,
        "class_acc": acc,
        "true_positives": true_positives,
        "true_negatives": true_negatives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "total_samples": total
    })
    print(f"Layer: {layer_name} | MSE: {avg_mse:.4f} | SSIM: {avg_ssim:.4f} | LPIPS: {avg_lpips:.4f} | Accuracy: {acc:.4f}")
    print(f"  Confusion Matrix - TP: {true_positives}, TN: {true_negatives}, FP: {false_positives}, FN: {false_negatives}")

# Write results to CSV
df = pd.DataFrame(results)
df.to_csv(METRICS_CSV, index=False)
print(f"Metrics written to {METRICS_CSV}")