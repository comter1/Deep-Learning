"""
Neural network backbones for feature extraction.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class SimpleCNN(nn.Module):
    """Simple CNN for MNIST/Fashion-MNIST."""
    
    def __init__(self, in_channels=1, latent_dim=128):
        super(SimpleCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        
        # After 3 pooling layers: 28x28 -> 14x14 -> 7x7 -> 3x3 (for MNIST 28x28)
        # For 128 channels: 128 * 3 * 3 = 1152
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, latent_dim)
        
    def forward(self, x):
        # Conv block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Conv block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Conv block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class ResNetBackbone(nn.Module):
    """ResNet backbone for CIFAR-10."""
    
    def __init__(self, latent_dim=128, pretrained=False):
        super(ResNetBackbone, self).__init__()
        
        # Load ResNet-50
        resnet = models.resnet50(pretrained=pretrained)
        
        # Remove the final fully connected layer
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        
        # Add a new FC layer to get desired latent dimension
        self.fc = nn.Linear(2048, latent_dim)
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x


def get_backbone(dataset_name, latent_dim, in_channels=1):
    """Factory function to get the appropriate backbone."""
    if dataset_name in ['mnist', 'fashion_mnist']:
        return SimpleCNN(in_channels=in_channels, latent_dim=latent_dim)
    elif dataset_name == 'cifar10':
        return ResNetBackbone(latent_dim=latent_dim)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
