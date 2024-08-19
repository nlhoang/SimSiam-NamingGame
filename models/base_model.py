import torch
import torch.nn as nn
import torch.nn.functional as F
from models.cifar_resnet import resnet18 as cifar_resnet
from torchvision.models import resnet18 as torchvision_resnet


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return x


class Encoder(nn.Module):
    def __init__(self, backbone='cnn'):
        super(Encoder, self).__init__()
        if backbone == 'resnet_cifar':
            self.resnet = cifar_resnet(pretrained=False, zero_init_residual=True)
            self.resnet.fc = nn.Identity()
            self.backbone = self.resnet
        elif backbone == 'resnet_torch':
            self.resnet = torchvision_resnet(pretrained=True)
            self.resnet.fc = nn.Identity()
            self.backbone = self.resnet
        else:  # backbone == 'cnn_3_64'
            self.backbone = CNN()

    def forward(self, x):
        x = self.backbone(x)
        return x


class Projector(nn.Module):
    def __init__(self, feature_dim, latent_dim):
        super(Projector, self).__init__()
        self.fc1 = nn.Linear(feature_dim, latent_dim)
        self.bn1 = nn.BatchNorm1d(latent_dim)
        self.fc2 = nn.Linear(latent_dim, latent_dim)
        self.bn2 = nn.BatchNorm1d(latent_dim)
        self.fc3 = nn.Linear(latent_dim, latent_dim)
        self.bn3 = nn.BatchNorm1d(latent_dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.bn3(self.fc3(x))
        return x
