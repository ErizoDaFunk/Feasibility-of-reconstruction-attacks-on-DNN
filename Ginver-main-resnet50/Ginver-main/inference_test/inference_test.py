import sys
import os
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import numpy as np

# Add parent directory to path to import Model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Model import Net

# Path to test image
IMAGE_PATH = "./test_images/test03.jpg"  # Update this path if needed

# ImageNet class labels
try:
    with open("imagenet_classes.txt") as f:
        classes = [line.strip() for line in f.readlines()]
except:
    print("Warning: imagenet_classes.txt not found. Will display only class index.")
    classes = None

def preprocess_image(image_path):
    """Preprocess the image as required by ResNet50"""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)  # Add batch dimension

def run_inference():
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = Net()
    model.eval()
    model.to(device)
    
    # Process image
    try:
        input_tensor = preprocess_image(IMAGE_PATH)
        input_tensor = input_tensor.to(device)
    except Exception as e:
        print(f"Error processing image: {e}")
        return
    
    # Run inference
    try:
        with torch.no_grad():
            output = model(input_tensor)
            
        # Get prediction
        _, predicted_idx = torch.max(output, 1)
        predicted_idx = predicted_idx.item()
        
        # Print results
        print(f"Predicted class index: {predicted_idx}")
        
        if classes:
            print(f"Predicted class: {classes[predicted_idx]}")
            
        # Print top 5 predictions
        prob = torch.nn.functional.softmax(output, dim=1)[0]
        top5_prob, top5_idx = torch.topk(prob, 5)
        
        print("\nTop 5 predictions:")
        for i, (p, idx) in enumerate(zip(top5_prob, top5_idx)):
            if classes:
                print(f"{i+1}. {classes[idx]}: {p.item():.4f}")
            else:
                print(f"{i+1}. Class {idx.item()}: {p.item():.4f}")
                
    except Exception as e:
        print(f"Inference error: {e}")

if __name__ == "__main__":
    run_inference()