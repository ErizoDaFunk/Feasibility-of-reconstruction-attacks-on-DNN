import sys
import os
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import numpy as np
from glob import glob
import torchvision.models as models
import warnings

# Add parent directory to path to import Model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Path to test images folder - update with your test image directory
TEST_IMAGES_PATH = "./test_images"  # Update this path if needed

# EMBL class labels
CLASSES = ['offsample', 'onsample']  # 0: offsample, 1: onsample

def preprocess_image(image_path):
    """Preprocess the image for the EMBL ResNet50 model"""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),  # ResNet50 expects 224x224 images
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet normalization
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Open image and convert to RGB (3 channels) as expected by ResNet50
    image = Image.open(image_path)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    return transform(image).unsqueeze(0)  # Add batch dimension

def run_inference():
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    try:
        # Create a much simpler approach - load the model directly without initializing
        model_path = "../../ModelResult/classifier/classifier.pth"
        model = torch.load(model_path, map_location=device)
        model.eval()
        print(f"Model loaded from {model_path}")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Failed to load model. Please check the model path and try again.")
        return  # Exit the function if model loading fails - don't try alternatives
    
    # Get all test images
    test_images = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        test_images.extend(glob(os.path.join(TEST_IMAGES_PATH, ext)))
    
    if not test_images:
        print(f"No images found in {TEST_IMAGES_PATH}")
        return
    
    print(f"Found {len(test_images)} test images")
    
    # Process each image
    results = []
    for img_path in test_images:
        try:
            print(f"\nProcessing image: {os.path.basename(img_path)}")
            input_tensor = preprocess_image(img_path)
            input_tensor = input_tensor.to(device)
            
            # Run inference
            with torch.no_grad():
                output = model(input_tensor)
                
            # Apply softmax to get probabilities
            prob = torch.softmax(output, dim=1)[0]
            
            # Get prediction
            predicted_idx = torch.argmax(prob).item()
            confidence = prob[predicted_idx].item()
            
            # Print results
            print(f"Predicted class: {CLASSES[predicted_idx]}")
            print(f"Confidence: {confidence:.4f}")
            
            # Store results for summary
            results.append({
                'image': os.path.basename(img_path),
                'prediction': CLASSES[predicted_idx],
                'confidence': confidence
            })
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            import traceback
            traceback.print_exc()
    
    # Print summary
    print("\n===== SUMMARY =====")
    print(f"Total images processed: {len(results)}")
    
    # Count predictions by class
    class_counts = {}
    for cls in CLASSES:
        count = sum(1 for r in results if r['prediction'] == cls)
        class_counts[cls] = count
        print(f"{cls}: {count} images")
    
    # Print high confidence predictions (>0.9)
    high_conf = [r for r in results if r['confidence'] > 0.9]
    print(f"\nHigh confidence predictions (>0.9): {len(high_conf)}")
    
if __name__ == "__main__":
    
    # Suppress warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    run_inference()