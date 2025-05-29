import torch
import torch.nn as nn
import torch.nn.functional as F
from base.cifar_resnet import resnet18 as cifar_resnet18
from base.cifar_resnet import resnet34 as cifar_resnet34
from torchvision.models import resnet18, resnet34, resnet50


class CNN_mnist(nn.Module):  # image of shape [1, 28, 28]
    def __init__(self):
        super(CNN_mnist, self).__init__()
        self.c = 16
        self.fc1 = nn.Conv2d(1, self.c, kernel_size=4, stride=2, padding=1)  # out: c x 14 x 14
        self.fc2 = nn.Conv2d(self.c, self.c*2, kernel_size=4, stride=2, padding=1)  # out: c x 7 x 7
        self.fc3 = nn.Linear(self.c*2*7*7, 512)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = x.view(x.size(0), -1)
        x = self.fc3(x)
        return x


class CNN_Img(nn.Module):  # image of shape [3, 64, 64]
    def __init__(self):
        super(CNN_Img, self).__init__()
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
    def __init__(self, backbone='cnn-img', freeze_backbone=False):
        super(Encoder, self).__init__()
        if backbone == 'resnet_cifar18':
            self.resnet = cifar_resnet18(pretrained=False, zero_init_residual=True)
            self.resnet.fc = nn.Identity()
            self.backbone = self.resnet
        elif backbone == 'resnet_cifar34':
            self.resnet = cifar_resnet34(pretrained=False, zero_init_residual=True)
            self.resnet.fc = nn.Identity()
            self.backbone = self.resnet
        elif backbone == 'resnet18':
            self.resnet = resnet18(pretrained=False, zero_init_residual=True)
            self.resnet.fc = nn.Identity()
            self.backbone = self.resnet
        elif backbone == 'resnet34':
            self.resnet = resnet34(pretrained=False, zero_init_residual=True)
            self.resnet.fc = nn.Identity()
            self.backbone = self.resnet
        elif backbone == 'resnet50':
            self.resnet = resnet50(pretrained=False, zero_init_residual=True)
            self.resnet.fc = nn.Identity()
            self.backbone = self.resnet
        elif backbone == 'cnn-img':
            self.backbone = CNN_Img()
        else:  # backbone == 'cnn-mnist'
            self.backbone = CNN_mnist()

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

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
