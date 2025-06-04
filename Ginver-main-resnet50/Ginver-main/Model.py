
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class Net(nn.Module):
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

        # ----------- LAYER 1 -----------#
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

        # ----------- LAYER 2 -----------#
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

        # ----------- LAYER 3 -----------#
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

        # ----------- LAYER 4 -----------#
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

        # ----------- FINAL LAYERS -----------#
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


class ResNet50EMBL(nn.Module):
    def __init__(self, model):
        super(ResNet50EMBL, self).__init__()

        # First layer components
        self.conv1 = model[0][0]   # 7×7, 64, stride 2
        self.bn1 = model[0][1]      # Batch normalization
        self.relu = model[0][2]    # ReLU activation
        self.maxpool = model[0][3]  # 3×3 max pool, stride 2

        # Layer1 - 3 Bottleneck blocks (total of 9 convolutional layers)
        layer1 = model[0][4]

        self.layer1_0_downsample = layer1[0].downsample
        self.layer1_0_conv1 = layer1[0].conv1  # 1×1, 64
        self.layer1_0_bn1 = layer1[0].bn1
        self.layer1_0_conv2 = layer1[0].conv2  # 3×3, 64
        self.layer1_0_bn2 = layer1[0].bn2
        self.layer1_0_conv3 = layer1[0].conv3  # 1×1, 256
        self.layer1_0_bn3 = layer1[0].bn3

        self.layer1_1_conv1 = layer1[1].conv1  # 1×1, 64
        self.layer1_1_bn1 = layer1[1].bn1
        self.layer1_1_conv2 = layer1[1].conv2  # 3×3, 64
        self.layer1_1_bn2 = layer1[1].bn2
        self.layer1_1_conv3 = layer1[1].conv3  # 1×1, 256
        self.layer1_1_bn3 = layer1[1].bn3

        self.layer1_2_conv1 = layer1[2].conv1  # 1×1, 64
        self.layer1_2_bn1 = layer1[2].bn1
        self.layer1_2_conv2 = layer1[2].conv2  # 3×3, 64
        self.layer1_2_bn2 = layer1[2].bn2
        self.layer1_2_conv3 = layer1[2].conv3  # 1×1, 256
        self.layer1_2_bn3 = layer1[2].bn3

        # Layer2 - 4 Bottleneck blocks (total of 12 convolutional layers)
        layer2 = model[0][5]

        self.layer2_0_downsample = layer2[0].downsample
        self.layer2_0_conv1 = layer2[0].conv1  # 1×1, 128
        self.layer2_0_bn1 = layer2[0].bn1
        self.layer2_0_conv2 = layer2[0].conv2  # 3×3, 128
        self.layer2_0_bn2 = layer2[0].bn2
        self.layer2_0_conv3 = layer2[0].conv3  # 1×1, 512
        self.layer2_0_bn3 = layer2[0].bn3

        self.layer2_1_conv1 = layer2[1].conv1  # 1×1, 128
        self.layer2_1_bn1 = layer2[1].bn1
        self.layer2_1_conv2 = layer2[1].conv2  # 3×3, 128
        self.layer2_1_bn2 = layer2[1].bn2
        self.layer2_1_conv3 = layer2[1].conv3  # 1×1, 512
        self.layer2_1_bn3 = layer2[1].bn3

        self.layer2_2_conv1 = layer2[2].conv1  # 1×1, 128
        self.layer2_2_bn1 = layer2[2].bn1
        self.layer2_2_conv2 = layer2[2].conv2  # 3×3, 128
        self.layer2_2_bn2 = layer2[2].bn2
        self.layer2_2_conv3 = layer2[2].conv3  # 1×1, 512
        self.layer2_2_bn3 = layer2[2].bn3

        self.layer2_3_conv1 = layer2[3].conv1  # 1×1, 128
        self.layer2_3_bn1 = layer2[3].bn1
        self.layer2_3_conv2 = layer2[3].conv2  # 3×3, 128
        self.layer2_3_bn2 = layer2[3].bn2
        self.layer2_3_conv3 = layer2[3].conv3  # 1×1, 512
        self.layer2_3_bn3 = layer2[3].bn3

        # Layer3 - 6 Bottleneck blocks (total of 18 convolutional layers)
        layer3 = model[0][6]

        self.layer3_0_downsample = layer3[0].downsample
        self.layer3_0_conv1 = layer3[0].conv1  # 1×1, 256
        self.layer3_0_bn1 = layer3[0].bn1
        self.layer3_0_conv2 = layer3[0].conv2  # 3×3, 256
        self.layer3_0_bn2 = layer3[0].bn2
        self.layer3_0_conv3 = layer3[0].conv3  # 1×1, 1024
        self.layer3_0_bn3 = layer3[0].bn3

        self.layer3_1_conv1 = layer3[1].conv1  # 1×1, 256
        self.layer3_1_bn1 = layer3[1].bn1
        self.layer3_1_conv2 = layer3[1].conv2  # 3×3, 256
        self.layer3_1_bn2 = layer3[1].bn2
        self.layer3_1_conv3 = layer3[1].conv3  # 1×1, 1024
        self.layer3_1_bn3 = layer3[1].bn3

        self.layer3_2_conv1 = layer3[2].conv1  # 1×1, 256
        self.layer3_2_bn1 = layer3[2].bn1
        self.layer3_2_conv2 = layer3[2].conv2  # 3×3, 256
        self.layer3_2_bn2 = layer3[2].bn2
        self.layer3_2_conv3 = layer3[2].conv3  # 1×1, 1024
        self.layer3_2_bn3 = layer3[2].bn3

        self.layer3_3_conv1 = layer3[3].conv1  # 1×1, 256
        self.layer3_3_bn1 = layer3[3].bn1
        self.layer3_3_conv2 = layer3[3].conv2  # 3×3, 256
        self.layer3_3_bn2 = layer3[3].bn2
        self.layer3_3_conv3 = layer3[3].conv3  # 1×1, 1024
        self.layer3_3_bn3 = layer3[3].bn3

        self.layer3_4_conv1 = layer3[4].conv1  # 1×1, 256
        self.layer3_4_bn1 = layer3[4].bn1
        self.layer3_4_conv2 = layer3[4].conv2  # 3×3, 256
        self.layer3_4_bn2 = layer3[4].bn2
        self.layer3_4_conv3 = layer3[4].conv3  # 1×1, 1024
        self.layer3_4_bn3 = layer3[4].bn3

        self.layer3_5_conv1 = layer3[5].conv1  # 1×1, 256
        self.layer3_5_bn1 = layer3[5].bn1
        self.layer3_5_conv2 = layer3[5].conv2  # 3×3, 256
        self.layer3_5_bn2 = layer3[5].bn2
        self.layer3_5_conv3 = layer3[5].conv3  # 1×1, 1024
        self.layer3_5_bn3 = layer3[5].bn3

        # Layer4 - 3 Bottleneck blocks (total of 9 convolutional layers)
        layer4 = model[0][7]

        self.layer4_0_downsample = layer4[0].downsample
        self.layer4_0_conv1 = layer4[0].conv1  # 1×1, 512
        self.layer4_0_bn1 = layer4[0].bn1
        self.layer4_0_conv2 = layer4[0].conv2  # 3×3, 512
        self.layer4_0_bn2 = layer4[0].bn2
        self.layer4_0_conv3 = layer4[0].conv3  # 1×1, 2048
        self.layer4_0_bn3 = layer4[0].bn3

        self.layer4_1_conv1 = layer4[1].conv1  # 1×1, 512
        self.layer4_1_bn1 = layer4[1].bn1
        self.layer4_1_conv2 = layer4[1].conv2  # 3×3, 512
        self.layer4_1_bn2 = layer4[1].bn2
        self.layer4_1_conv3 = layer4[1].conv3  # 1×1, 2048
        self.layer4_1_bn3 = layer4[1].bn3

        self.layer4_2_conv1 = layer4[2].conv1  # 1×1, 512
        self.layer4_2_bn1 = layer4[2].bn1
        self.layer4_2_conv2 = layer4[2].conv2  # 3×3, 512
        self.layer4_2_bn2 = layer4[2].bn2
        self.layer4_2_conv3 = layer4[2].conv3  # 1×1, 2048
        self.layer4_2_bn3 = layer4[2].bn3

        # Final layers
        final_layers = model[1]
        self.final_layers_adaptive_concat_pool_2d = final_layers[0]  # Adaptive average pooling

        # Fully connected 1
        self.final_layers_flatten = final_layers[1]
        self.final_layers_bn1 = final_layers[2]  # Batch normalization
        self.final_layers_dropout_1 = final_layers[3]  # Dropout
        self.final_layers_linear_1 = final_layers[4]  # Linear layer (2 classes for EMBL)

        # Fully connected 2
        self.final_layers_bn2 = final_layers[6]  # Batch normalization
        self.final_layers_dropout_2 = final_layers[7]  # Dropout
        self.final_layers_linear_2 = final_layers[8]  # Linear layer (2 classes for EMBL)

        # For convenience, also keep the original layer blocks
        self.layer1 = layer1
        self.layer2 = layer2
        self.layer3 = layer3
        self.layer4 = layer4

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

        # ----------- LAYER 1 -----------#
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

        # ----------- LAYER 2 -----------#
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

        # ----------- LAYER 3 -----------#
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

        # ----------- LAYER 4 -----------#
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

        # ----------- FINAL LAYERS -----------#
        # Adaptive Average Pooling
        x = self.final_layers_adaptive_concat_pool_2d(x)
        if layer_name == 'adaptive_concat_pool':
            return x

        # Flatten and fully connected layer
        x = self.final_layers_flatten(x)

        x = self.final_layers_bn1(x)
        x = self.final_layers_dropout_1(x)
        x = self.final_layers_linear_1(x)

        if layer_name == 'fc_1':
            return x

        x = F.relu(x)

        x = self.final_layers_bn2(x)
        x = self.final_layers_dropout_2(x)
        x = self.final_layers_linear_2(x)

        if layer_name == 'fc_2':
            return x

        # Default: return the final output with softmax
        return x

    def eval(self):
        # Llama al método eval() de la clase base (nn.Module)
        super(ResNet50EMBL, self).eval()
        return self  # Devuelve self para permitir encadenamiento

################################################################################
#                                                                              #
#   ResNet Inversion Model Arquitectures depending of the number of layers     #
#                                                                              #
################################################################################

class ResNetInversion_Conv1(nn.Module):
    def __init__(self, nc=3, ngf=64):
        super(ResNetInversion_Conv1, self).__init__()
        
        self.decoder = nn.Sequential(
            # Invert Conv1 (7×7, 64, stride 2)
            nn.ConvTranspose2d(
                in_channels=64,         # Input features from conv1
                out_channels=nc,        # Output channels (RGB image)
                kernel_size=7,          # Same as original
                stride=2,               # Same as original
                padding=3,              # Same as original
                output_padding=1,       # Needed for odd dimensions
                bias=False              # Typically no bias in ResNet conv1
            ),
            
            # Final activation to produce valid image values
            nn.Tanh()
        )
    
    def forward(self, x):
        return self.decoder(x)
    
class ResNetInversion_bn1(nn.Module):
    def __init__(self, nc=3, ngf=64):
        super(ResNetInversion_bn1, self).__init__()
        
        self.decoder = nn.Sequential(

            # Invert BatchNorm - just use another BatchNorm with learned parameters
            nn.BatchNorm2d(64),
            
            # Invert Conv (7×7, 64, stride 2) - transpose convolution
            nn.ConvTranspose2d(64, nc, kernel_size=7, stride=2, padding=3, output_padding=1),
            
            # Final activation to produce image-like outputs
            nn.Tanh()
        )
    
    def forward(self, x):
        return self.decoder(x)


class ResNetInversion_MaxPool(nn.Module):
    def __init__(self, nc=3, ngf=64):
        super(ResNetInversion_MaxPool, self).__init__()
        
        self.decoder = nn.Sequential(
            # 1. Invert MaxPool (3×3, stride 2)
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            
            # 2. No specific inversion for ReLU, but use ReLU in decoder
            nn.ReLU(inplace=True),
            
            # 3. Invert BatchNorm - just use another BatchNorm with learned parameters
            nn.BatchNorm2d(64),
            
            # 4. Invert Conv (7×7, 64, stride 2) - transpose convolution
            nn.ConvTranspose2d(64, nc, kernel_size=7, stride=2, padding=3, output_padding=1),
            
            # Final activation to produce image-like outputs
            nn.Tanh()
        )
    
    def forward(self, x):
        return self.decoder(x)
    

################################################################################
#                                                                              #
#           ResNet Inversion Model Arquitecture proposed by Paper              #
#                                                                              #
################################################################################

class ResnetInversion_Generic(nn.Module):
    # nc 要生成的图片通道 ngf 中间值 nz 输入的通道数
    def __init__(self, nc, ngf, nz):
        super(ResnetInversion_Generic, self).__init__()

        self.nc = nc
        self.ngf = ngf
        self.nz = nz

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(nz, 8*ngf, 4, 2, 1),
            nn.BatchNorm2d(8*ngf),
            nn.Tanh(),

            nn.ConvTranspose2d(8 * ngf, 4 * ngf, 4, 2, 1), #28*28
            nn.BatchNorm2d(4 * ngf),
            nn.Tanh(),

            nn.ConvTranspose2d(4 * ngf, 2 * ngf, 4, 2, 1), #56*56
            nn.BatchNorm2d(2 * ngf),
            nn.Tanh(),

            nn.ConvTranspose2d(2 * ngf, ngf, 4, 2, 1), #112*112
            nn.BatchNorm2d(ngf),
            nn.Tanh(),

            nn.ConvTranspose2d(ngf, nc, 4, 2, 1), #224*224
            nn.Sigmoid()
        )

    # def forward(self, x):
    #     # 这个nz应该是1024
    #     print(f"Input shape: {x.shape}")  # Debugging line to check input shape
    #     x = x.view(-1, self.nz, 7, 7)
    #     print(f"Reshaped input: {x.shape}")
    #     x = self.decoder(x)
    #     print(f"Output shape: {x.shape}")
    #     return x

    def forward(self, x):
        # Get original batch size
        batch_size = x.size(0)
        
        # Check feature map size to adapt reshaping
        if x.dim() == 2:  # Already flattened
            x = x.view(batch_size, self.nz, 7, 7)
        elif x.dim() == 4:  # Has spatial dimensions
            # Get current spatial dimensions
            _, _, h, w = x.size()
            
            # If spatial dimensions need adjustment, do it properly
            if h != 7 or w != 7:
                # Resize using adaptive pooling to preserve batch size
                x = F.adaptive_avg_pool2d(x, (7, 7))
        
        # Ensure batch size is preserved
        x = x.view(batch_size, self.nz, 7, 7)
        
        return self.decoder(x)