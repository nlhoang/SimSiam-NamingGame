import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils import param_count, CIFAR10Dataset, display_loss
from models.base_model import Encoder, Projector
from models.SimSiamVAE import PredictorDecoder, train


class PredictorEncoder(nn.Module):
    def __init__(self, latent_dim, variable_dim):
        super(PredictorEncoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, latent_dim // 2)
        self.bn1 = nn.BatchNorm1d(latent_dim // 2)
        self.fc_mu = nn.Linear(latent_dim // 2, variable_dim)
        self.fc_logvar = nn.Linear(latent_dim // 2, variable_dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        w = self.reparameterize(mu, logvar)
        return w, mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = torch.empty_like(std).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu


class SimSiamVAE(nn.Module):
    def __init__(self, feature_dim, latent_dim, variable_dim, backbone='resnet_torch'):
        super(SimSiamVAE, self).__init__()
        self.encoder = Encoder(backbone=backbone)
        self.projector = Projector(feature_dim=feature_dim, latent_dim=latent_dim)
        self.predictorEncoder = PredictorEncoder(latent_dim=latent_dim, variable_dim=variable_dim)
        self.predictorDecoder = PredictorDecoder(variable_dim=variable_dim, latent_dim=latent_dim)

    def forward(self, x1, x2):
        y1 = self.encoder(x1)
        y2 = self.encoder(x2)
        z1 = self.projector(y1)
        z2 = self.projector(y2)
        w1, mu1, logvar1 = self.predictorEncoder(z1)
        w2, mu2, logvar2 = self.predictorEncoder(z2)
        p1 = self.predictorDecoder(w1)
        p2 = self.predictorDecoder(w2)
        return z1, z2, mu1, logvar1, mu2, logvar2, p1, p2


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
    print(f"Using device: {device}")
    train_dataset = CIFAR10Dataset(path='../../data/', train=True)
    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

    # Training
    model = SimSiamVAE(feature_dim=512, latent_dim=256, variable_dim=512, backbone='cnn').to(device)
    print(model)
    print('Model Size: {}'.format(param_count(model)))
    loss_history = train(model, train_dataloader, learning_rate=0.001, device=device, epochs=500, save_interval=50, save_prefix='SimSiamVAE_CIFAR')
    display_loss(loss_history)
