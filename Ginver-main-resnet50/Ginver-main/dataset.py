import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset

class CIFAR10Dataset(Dataset):
    def __init__(self, train=True, transform=None):
        self.dataset = torchvision.datasets.CIFAR10(root='./data', train=train, download=True, transform=transform)

    def __getitem__(self, index):
        img, label = self.dataset[index]
        return img, label

    def __len__(self):
        return len(self.dataset)