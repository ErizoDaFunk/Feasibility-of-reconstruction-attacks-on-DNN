import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1) 
        ################################
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x, conv=-1, relu=-1):

        x = self.conv1(x)
        if conv == 1:
            return x
        
        x = F.relu(x)
        if relu == 1:
            return x
        
        x = self.conv2(x)
        if conv == 2:
            return x
        
        x = F.relu(x)
        if relu == 2:
            return x
        
        #########################
        x = F.max_pool2d(x, 2)

        x = self.dropout1(x)

        x = torch.flatten(x, 1)

        x = self.fc1(x)

        x = F.relu(x)
        if relu == 3:
            return x
        
        x = self.dropout2(x)

        x = self.fc2(x)

        output = F.log_softmax(x, dim=1)

        return output
    
class AdversarialNet(nn.Module):
    def __init__(self, nc, ngf, nz = None):
        # nc number of channes - 3 colors
        # ngf - number of generator filters - to be defined
        # nz - latent space dimension - to be defined
        super(AdversarialNet, self).__init__()

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, 1),
            nn.ConvTranspose2d(32, 1, 3, 1)
        )


    def forward(self, x):
        # x = x.view(-1, self.nz, 64, 64)
        x = self.decoder(x)
        return x