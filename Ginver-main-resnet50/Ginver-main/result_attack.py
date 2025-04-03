import os
import torch # type: ignore
from Model import Net, AdversarialNet
from torchvision import datasets, transforms # type: ignore
import matplotlib.pyplot as plt # type: ignore
import numpy as np
from skimage.metrics import structural_similarity as ssim # type: ignore

# Establecer la variable de entorno para evitar el error de OpenMP
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Configuración del dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cargar el modelo clasificador
classifier = Net().to(device)
classifier.load_state_dict(torch.load("../ModelResult/classifier/classifier_32.pth", map_location=device, weights_only=True ))
classifier.eval()

# Cargar el modelo de ataque
inversion = AdversarialNet(nc=1, ngf=128, nz=128).to(device)
checkpoint = torch.load('../ModelResult/blackbox/train/relu2_all_white_64/final_inversion.pth', map_location=device, weights_only=True)
inversion.load_state_dict(checkpoint['model'])
inversion.eval()

# Preparar el conjunto de datos de prueba
transform = transforms.Compose([
    transforms.ToTensor(),
])
test_set = datasets.MNIST('../data', train=False, transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True)

# Generar y mostrar imágenes invertidas
fig, axes = plt.subplots(5, 3, figsize=(15, 15))

for i, (data, target) in enumerate(test_loader):
    if i >= 5:
        break
    
    data, target = data.to(device), target.to(device)
    
    # Obtener la salida del clasificador
    with torch.no_grad():
        output = classifier(data, relu=2)
    
    # Generar la imagen invertida
    with torch.no_grad():
        inverted_image = inversion(output)
    
    # Convertir las imágenes a formato numpy
    original_image = data.cpu().numpy().squeeze()
    inverted_image = inverted_image.cpu().numpy().squeeze()
    
    # Calcular MSE
    mse_value = np.mean((original_image - inverted_image) ** 2)
    
    # Calcular SSIM
    ssim_value = ssim(original_image, inverted_image, data_range=original_image.max() - original_image.min())
    
    # Mostrar la imagen original
    axes[i, 0].imshow(original_image, cmap='gray')
    axes[i, 0].set_title(f'Original Image {i+1}')
    axes[i, 0].axis('off')
    
    # Mostrar la imagen invertida
    axes[i, 1].imshow(inverted_image, cmap='gray')
    axes[i, 1].set_title(f'Reconstructed Image {i+1}')
    axes[i, 1].axis('off')
    
    # Mostrar las métricas
    axes[i, 2].text(0.5, 0.5, f'MSE: {mse_value:.4f}\nSSIM: {ssim_value:.4f}', 
                    horizontalalignment='center', verticalalignment='center', fontsize=12)
    axes[i, 2].axis('off')

plt.tight_layout()
plt.show()