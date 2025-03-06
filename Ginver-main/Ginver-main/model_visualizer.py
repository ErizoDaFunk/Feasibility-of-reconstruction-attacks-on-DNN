from Model import Classifier
import torch
from netron_export import export_model

nc = 1
nz = 530

classifier = Classifier(nc=nc, ndf=128, nz=nz)

# Input size: torch.Size([128, 1, 64, 64]) Output size: torch.Size([128, 530]), Target size: torch.Size([128])

x = torch.rand(1, 1, 64, 64)
output = model(x)

