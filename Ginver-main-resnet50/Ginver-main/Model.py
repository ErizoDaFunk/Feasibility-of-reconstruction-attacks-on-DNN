import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class Net(nn.Module):

    # def __init__(self):
    #     super(Net, self).__init__()
    #     resnet = models.resnet50(pretrained=True)
    #     self.features = nn.Sequential(*list(resnet.children())[:-1])  # All layers except FC
    #     self.fc = resnet.fc  # Original FC layer

    # def forward(self, x):
    #     x = self.features(x)
    #     x = torch.flatten(x, 1)
    #     x = self.fc(x)
    #     return x  # Return logits without applying softmax

    def __init__(self):
        super(Net, self).__init__()
        # Load pre-trained ResNet50
        resnet = models.resnet50(pretrained=True)

        # First layer components
        self.conv1 = resnet.conv1  # 7×7, 64, stride 2
        self.bn1 = resnet.bn1      # Batch normalization
        self.relu = resnet.relu    # ReLU activation
        self.maxpool = resnet.maxpool  # 3×3 max pool, stride 2

        # Layer1 - 3 Bottleneck blocks (total of 9 convolutional layers)
        self.layer1_0_downsample = resnet.layer1[0].downsample
        self.layer1_0_conv1 = resnet.layer1[0].conv1  # 1×1, 64
        self.layer1_0_bn1 = resnet.layer1[0].bn1
        self.layer1_0_conv2 = resnet.layer1[0].conv2  # 3×3, 64
        self.layer1_0_bn2 = resnet.layer1[0].bn2
        self.layer1_0_conv3 = resnet.layer1[0].conv3  # 1×1, 256
        self.layer1_0_bn3 = resnet.layer1[0].bn3
        
        self.layer1_1_conv1 = resnet.layer1[1].conv1  # 1×1, 64
        self.layer1_1_bn1 = resnet.layer1[1].bn1
        self.layer1_1_conv2 = resnet.layer1[1].conv2  # 3×3, 64
        self.layer1_1_bn2 = resnet.layer1[1].bn2
        self.layer1_1_conv3 = resnet.layer1[1].conv3  # 1×1, 256
        self.layer1_1_bn3 = resnet.layer1[1].bn3
        
        self.layer1_2_conv1 = resnet.layer1[2].conv1  # 1×1, 64
        self.layer1_2_bn1 = resnet.layer1[2].bn1
        self.layer1_2_conv2 = resnet.layer1[2].conv2  # 3×3, 64
        self.layer1_2_bn2 = resnet.layer1[2].bn2
        self.layer1_2_conv3 = resnet.layer1[2].conv3  # 1×1, 256
        self.layer1_2_bn3 = resnet.layer1[2].bn3

        # Layer2 - 4 Bottleneck blocks (total of 12 convolutional layers)
        self.layer2_0_downsample = resnet.layer2[0].downsample
        self.layer2_0_conv1 = resnet.layer2[0].conv1  # 1×1, 128
        self.layer2_0_bn1 = resnet.layer2[0].bn1
        self.layer2_0_conv2 = resnet.layer2[0].conv2  # 3×3, 128
        self.layer2_0_bn2 = resnet.layer2[0].bn2
        self.layer2_0_conv3 = resnet.layer2[0].conv3  # 1×1, 512
        self.layer2_0_bn3 = resnet.layer2[0].bn3
        
        self.layer2_1_conv1 = resnet.layer2[1].conv1  # 1×1, 128
        self.layer2_1_bn1 = resnet.layer2[1].bn1
        self.layer2_1_conv2 = resnet.layer2[1].conv2  # 3×3, 128
        self.layer2_1_bn2 = resnet.layer2[1].bn2
        self.layer2_1_conv3 = resnet.layer2[1].conv3  # 1×1, 512
        self.layer2_1_bn3 = resnet.layer2[1].bn3
        
        self.layer2_2_conv1 = resnet.layer2[2].conv1  # 1×1, 128
        self.layer2_2_bn1 = resnet.layer2[2].bn1
        self.layer2_2_conv2 = resnet.layer2[2].conv2  # 3×3, 128
        self.layer2_2_bn2 = resnet.layer2[2].bn2
        self.layer2_2_conv3 = resnet.layer2[2].conv3  # 1×1, 512
        self.layer2_2_bn3 = resnet.layer2[2].bn3
        
        self.layer2_3_conv1 = resnet.layer2[3].conv1  # 1×1, 128
        self.layer2_3_bn1 = resnet.layer2[3].bn1
        self.layer2_3_conv2 = resnet.layer2[3].conv2  # 3×3, 128
        self.layer2_3_bn2 = resnet.layer2[3].bn2
        self.layer2_3_conv3 = resnet.layer2[3].conv3  # 1×1, 512
        self.layer2_3_bn3 = resnet.layer2[3].bn3

        # Layer3 - 6 Bottleneck blocks (total of 18 convolutional layers)
        self.layer3_0_downsample = resnet.layer3[0].downsample
        self.layer3_0_conv1 = resnet.layer3[0].conv1  # 1×1, 256
        self.layer3_0_bn1 = resnet.layer3[0].bn1
        self.layer3_0_conv2 = resnet.layer3[0].conv2  # 3×3, 256
        self.layer3_0_bn2 = resnet.layer3[0].bn2
        self.layer3_0_conv3 = resnet.layer3[0].conv3  # 1×1, 1024
        self.layer3_0_bn3 = resnet.layer3[0].bn3
        
        self.layer3_1_conv1 = resnet.layer3[1].conv1  # 1×1, 256
        self.layer3_1_bn1 = resnet.layer3[1].bn1
        self.layer3_1_conv2 = resnet.layer3[1].conv2  # 3×3, 256
        self.layer3_1_bn2 = resnet.layer3[1].bn2
        self.layer3_1_conv3 = resnet.layer3[1].conv3  # 1×1, 1024
        self.layer3_1_bn3 = resnet.layer3[1].bn3
        
        self.layer3_2_conv1 = resnet.layer3[2].conv1  # 1×1, 256
        self.layer3_2_bn1 = resnet.layer3[2].bn1
        self.layer3_2_conv2 = resnet.layer3[2].conv2  # 3×3, 256
        self.layer3_2_bn2 = resnet.layer3[2].bn2
        self.layer3_2_conv3 = resnet.layer3[2].conv3  # 1×1, 1024
        self.layer3_2_bn3 = resnet.layer3[2].bn3
        
        self.layer3_3_conv1 = resnet.layer3[3].conv1  # 1×1, 256
        self.layer3_3_bn1 = resnet.layer3[3].bn1
        self.layer3_3_conv2 = resnet.layer3[3].conv2  # 3×3, 256
        self.layer3_3_bn2 = resnet.layer3[3].bn2
        self.layer3_3_conv3 = resnet.layer3[3].conv3  # 1×1, 1024
        self.layer3_3_bn3 = resnet.layer3[3].bn3
        
        self.layer3_4_conv1 = resnet.layer3[4].conv1  # 1×1, 256
        self.layer3_4_bn1 = resnet.layer3[4].bn1
        self.layer3_4_conv2 = resnet.layer3[4].conv2  # 3×3, 256
        self.layer3_4_bn2 = resnet.layer3[4].bn2
        self.layer3_4_conv3 = resnet.layer3[4].conv3  # 1×1, 1024
        self.layer3_4_bn3 = resnet.layer3[4].bn3
        
        self.layer3_5_conv1 = resnet.layer3[5].conv1  # 1×1, 256
        self.layer3_5_bn1 = resnet.layer3[5].bn1
        self.layer3_5_conv2 = resnet.layer3[5].conv2  # 3×3, 256
        self.layer3_5_bn2 = resnet.layer3[5].bn2
        self.layer3_5_conv3 = resnet.layer3[5].conv3  # 1×1, 1024
        self.layer3_5_bn3 = resnet.layer3[5].bn3

        # Layer4 - 3 Bottleneck blocks (total of 9 convolutional layers)
        self.layer4_0_downsample = resnet.layer4[0].downsample
        self.layer4_0_conv1 = resnet.layer4[0].conv1  # 1×1, 512
        self.layer4_0_bn1 = resnet.layer4[0].bn1
        self.layer4_0_conv2 = resnet.layer4[0].conv2  # 3×3, 512
        self.layer4_0_bn2 = resnet.layer4[0].bn2
        self.layer4_0_conv3 = resnet.layer4[0].conv3  # 1×1, 2048
        self.layer4_0_bn3 = resnet.layer4[0].bn3
        
        self.layer4_1_conv1 = resnet.layer4[1].conv1  # 1×1, 512
        self.layer4_1_bn1 = resnet.layer4[1].bn1
        self.layer4_1_conv2 = resnet.layer4[1].conv2  # 3×3, 512
        self.layer4_1_bn2 = resnet.layer4[1].bn2
        self.layer4_1_conv3 = resnet.layer4[1].conv3  # 1×1, 2048
        self.layer4_1_bn3 = resnet.layer4[1].bn3
        
        self.layer4_2_conv1 = resnet.layer4[2].conv1  # 1×1, 512
        self.layer4_2_bn1 = resnet.layer4[2].bn1
        self.layer4_2_conv2 = resnet.layer4[2].conv2  # 3×3, 512
        self.layer4_2_bn2 = resnet.layer4[2].bn2
        self.layer4_2_conv3 = resnet.layer4[2].conv3  # 1×1, 2048
        self.layer4_2_bn3 = resnet.layer4[2].bn3

        # Final layers
        self.avgpool = resnet.avgpool  # Global average pooling
        self.fc = resnet.fc  # Fully connected layer (1000 classes for ImageNet)

        # For convenience, also keep the original layer blocks
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4


    def forward(self, x, layer_name=None):
        """
        Forward pass with option to return intermediate layer activations
        
        Args:
            x: Input tensor (batch of images)
            layer_name: Name of the layer to return activations from
        
        Returns:
            Tensor of activations from specified layer or final output
        """
        # Initial layers
        x = self.conv1(x)
        if layer_name == 'conv1':
            return x
        
        x = self.bn1(x)
        x = self.relu(x)
        if layer_name == 'relu1':
            return x
        
        x = self.maxpool(x)
        if layer_name == 'maxpool':
            return x
        
        #----------- LAYER 1 -----------#
        # Layer 1 - Block 0
        identity = x
        if self.layer1_0_downsample is not None:
            identity = self.layer1_0_downsample(identity)
        
        x = self.layer1_0_conv1(x)
        x = self.layer1_0_bn1(x)
        x = F.relu(x)
        
        x = self.layer1_0_conv2(x)
        x = self.layer1_0_bn2(x)
        x = F.relu(x)
        
        x = self.layer1_0_conv3(x)
        x = self.layer1_0_bn3(x)
        
        x += identity
        x = F.relu(x)
        if layer_name == 'layer1_0':
            return x
        
        # Layer 1 - Block 1
        identity = x
        
        x = self.layer1_1_conv1(x)
        x = self.layer1_1_bn1(x)
        x = F.relu(x)
        
        x = self.layer1_1_conv2(x)
        x = self.layer1_1_bn2(x)
        x = F.relu(x)
        
        x = self.layer1_1_conv3(x)
        x = self.layer1_1_bn3(x)
        
        x += identity
        x = F.relu(x)
        if layer_name == 'layer1_1':
            return x
        
        # Layer 1 - Block 2
        identity = x
        
        x = self.layer1_2_conv1(x)
        x = self.layer1_2_bn1(x)
        x = F.relu(x)
        
        x = self.layer1_2_conv2(x)
        x = self.layer1_2_bn2(x)
        x = F.relu(x)
        
        x = self.layer1_2_conv3(x)
        x = self.layer1_2_bn3(x)
        
        x += identity
        x = F.relu(x)
        if layer_name == 'layer1_2' or layer_name == 'layer1':
            return x
        
        #----------- LAYER 2 -----------#
        # Layer 2 - Block 0
        identity = x
        if self.layer2_0_downsample is not None:
            identity = self.layer2_0_downsample(identity)
        
        x = self.layer2_0_conv1(x)
        x = self.layer2_0_bn1(x)
        x = F.relu(x)
        
        x = self.layer2_0_conv2(x)
        x = self.layer2_0_bn2(x)
        x = F.relu(x)
        
        x = self.layer2_0_conv3(x)
        x = self.layer2_0_bn3(x)
        
        x += identity
        x = F.relu(x)
        if layer_name == 'layer2_0':
            return x
        
        # Layer 2 - Block 1
        identity = x
        
        x = self.layer2_1_conv1(x)
        x = self.layer2_1_bn1(x)
        x = F.relu(x)
        
        x = self.layer2_1_conv2(x)
        x = self.layer2_1_bn2(x)
        x = F.relu(x)
        
        x = self.layer2_1_conv3(x)
        x = self.layer2_1_bn3(x)
        
        x += identity
        x = F.relu(x)
        if layer_name == 'layer2_1':
            return x
        
        # Layer 2 - Block 2
        identity = x
        
        x = self.layer2_2_conv1(x)
        x = self.layer2_2_bn1(x)
        x = F.relu(x)
        
        x = self.layer2_2_conv2(x)
        x = self.layer2_2_bn2(x)
        x = F.relu(x)
        
        x = self.layer2_2_conv3(x)
        x = self.layer2_2_bn3(x)
        
        x += identity
        x = F.relu(x)
        if layer_name == 'layer2_2':
            return x
        
        # Layer 2 - Block 3
        identity = x
        
        x = self.layer2_3_conv1(x)
        x = self.layer2_3_bn1(x)
        x = F.relu(x)
        
        x = self.layer2_3_conv2(x)
        x = self.layer2_3_bn2(x)
        x = F.relu(x)
        
        x = self.layer2_3_conv3(x)
        x = self.layer2_3_bn3(x)
        
        x += identity
        x = F.relu(x)
        if layer_name == 'layer2_3' or layer_name == 'layer2':
            return x
        
        #----------- LAYER 3 -----------#
        # Layer 3 - Block 0
        identity = x
        if self.layer3_0_downsample is not None:
            identity = self.layer3_0_downsample(identity)
        
        x = self.layer3_0_conv1(x)
        x = self.layer3_0_bn1(x)
        x = F.relu(x)
        
        x = self.layer3_0_conv2(x)
        x = self.layer3_0_bn2(x)
        x = F.relu(x)
        
        x = self.layer3_0_conv3(x)
        x = self.layer3_0_bn3(x)
        
        x += identity
        x = F.relu(x)
        if layer_name == 'layer3_0':
            return x
        
        # Layer 3 - Block 1
        identity = x
        
        x = self.layer3_1_conv1(x)
        x = self.layer3_1_bn1(x)
        x = F.relu(x)
        
        x = self.layer3_1_conv2(x)
        x = self.layer3_1_bn2(x)
        x = F.relu(x)
        
        x = self.layer3_1_conv3(x)
        x = self.layer3_1_bn3(x)
        
        x += identity
        x = F.relu(x)
        if layer_name == 'layer3_1':
            return x
        
        # Layer 3 - Block 2
        identity = x
        
        x = self.layer3_2_conv1(x)
        x = self.layer3_2_bn1(x)
        x = F.relu(x)
        
        x = self.layer3_2_conv2(x)
        x = self.layer3_2_bn2(x)
        x = F.relu(x)
        
        x = self.layer3_2_conv3(x)
        x = self.layer3_2_bn3(x)
        
        x += identity
        x = F.relu(x)
        if layer_name == 'layer3_2':
            return x
        
        # Layer 3 - Block 3
        identity = x
        
        x = self.layer3_3_conv1(x)
        x = self.layer3_3_bn1(x)
        x = F.relu(x)
        
        x = self.layer3_3_conv2(x)
        x = self.layer3_3_bn2(x)
        x = F.relu(x)
        
        x = self.layer3_3_conv3(x)
        x = self.layer3_3_bn3(x)
        
        x += identity
        x = F.relu(x)
        if layer_name == 'layer3_3':
            return x
        
        # Layer 3 - Block 4
        identity = x
        
        x = self.layer3_4_conv1(x)
        x = self.layer3_4_bn1(x)
        x = F.relu(x)
        
        x = self.layer3_4_conv2(x)
        x = self.layer3_4_bn2(x)
        x = F.relu(x)
        
        x = self.layer3_4_conv3(x)
        x = self.layer3_4_bn3(x)
        
        x += identity
        x = F.relu(x)
        if layer_name == 'layer3_4':
            return x
        
        # Layer 3 - Block 5
        identity = x
        
        x = self.layer3_5_conv1(x)
        x = self.layer3_5_bn1(x)
        x = F.relu(x)
        
        x = self.layer3_5_conv2(x)
        x = self.layer3_5_bn2(x)
        x = F.relu(x)
        
        x = self.layer3_5_conv3(x)
        x = self.layer3_5_bn3(x)
        
        x += identity
        x = F.relu(x)
        if layer_name == 'layer3_5' or layer_name == 'layer3':
            return x
        
        #----------- LAYER 4 -----------#
        # Layer 4 - Block 0
        identity = x
        if self.layer4_0_downsample is not None:
            identity = self.layer4_0_downsample(identity)
        
        x = self.layer4_0_conv1(x)
        x = self.layer4_0_bn1(x)
        x = F.relu(x)
        
        x = self.layer4_0_conv2(x)
        x = self.layer4_0_bn2(x)
        x = F.relu(x)
        
        x = self.layer4_0_conv3(x)
        x = self.layer4_0_bn3(x)
        
        x += identity
        x = F.relu(x)
        if layer_name == 'layer4_0':
            return x
        
        # Layer 4 - Block 1
        identity = x
        
        x = self.layer4_1_conv1(x)
        x = self.layer4_1_bn1(x)
        x = F.relu(x)
        
        x = self.layer4_1_conv2(x)
        x = self.layer4_1_bn2(x)
        x = F.relu(x)
        
        x = self.layer4_1_conv3(x)
        x = self.layer4_1_bn3(x)
        
        x += identity
        x = F.relu(x)
        if layer_name == 'layer4_1':
            return x
        
        # Layer 4 - Block 2
        identity = x
        
        x = self.layer4_2_conv1(x)
        x = self.layer4_2_bn1(x)
        x = F.relu(x)
        
        x = self.layer4_2_conv2(x)
        x = self.layer4_2_bn2(x)
        x = F.relu(x)
        
        x = self.layer4_2_conv3(x)
        x = self.layer4_2_bn3(x)
        
        x += identity
        x = F.relu(x)
        if layer_name == 'layer4_2' or layer_name == 'layer4':
            return x
        
        #----------- FINAL LAYERS -----------#
        # Global Average Pooling
        x = self.avgpool(x)
        if layer_name == 'avgpool':
            return x
        
        # Flatten and fully connected layer
        x = torch.flatten(x, 1)
        x = self.fc(x)
        if layer_name == 'fc':
            return x
        
        # Default: return the final output with softmax
        return x
    

################################################################################
#                                                                              #
#   ResNet Inversion Model Arquitectures depending of the number of layers     #
#                                                                              #
################################################################################

class ResNetInversion_MaxPool(nn.Module):
    def __init__(self, nc=3, ngf=64):
        super(ResNetInversion_MaxPool, self).__init__()

        
        self.decoder = nn.Sequential(

        )
    
    def forward(self, x):
        return self.decoder(x)