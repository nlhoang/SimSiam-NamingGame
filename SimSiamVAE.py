import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from power_spherical import PowerSpherical
from base.utils import param_count, display_loss
from base.base_model import Encoder, Projector
from base.dataloader import FashionMNISTDataset


class PredictorEncoder(nn.Module):
    def __init__(self, latent_dim, variable_dim):
        super(PredictorEncoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, latent_dim // 2)
        self.bn1 = nn.BatchNorm1d(latent_dim // 2)
        self.fc2 = nn.Linear(latent_dim // 2, variable_dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.fc2(x)
        return x


class PredictorDecoder(nn.Module):
    def __init__(self, variable_dim, latent_dim):
        super(PredictorDecoder, self).__init__()
        self.fc1 = nn.Linear(variable_dim, latent_dim // 2)
        self.bn1 = nn.BatchNorm1d(latent_dim // 2)
        self.fc2 = nn.Linear(latent_dim // 2, latent_dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = torch.sigmoid(self.fc2(x))
        return x


class SimSiamVAE(nn.Module):
    def __init__(self, feature_dim, latent_dim, variable_dim, backbone='resnet-torch', freeze_backbone=False):
        super(SimSiamVAE, self).__init__()
        self.encoder = Encoder(backbone=backbone, freeze_backbone=freeze_backbone)
        self.projector = Projector(feature_dim=feature_dim, latent_dim=latent_dim)
        self.predictor_mu = PredictorEncoder(latent_dim=latent_dim, variable_dim=variable_dim*2)
        self.predictor_kappa = PredictorEncoder(latent_dim=latent_dim, variable_dim=1)
        self.decoder = PredictorDecoder(variable_dim=variable_dim*2, latent_dim=latent_dim)

    def forward(self, x1, x2):
        y1 = self.encoder(x1)
        y2 = self.encoder(x2)
        z1 = self.projector(y1)
        z2 = self.projector(y2)
        mu1 = F.normalize(self.predictor_mu(z1), dim=-1)  # Unit norm
        mu2 = F.normalize(self.predictor_mu(z2), dim=-1)
        kappa1 = F.softplus(self.predictor_kappa(z1)) + 1e-6  # Ensure kappa is positive
        kappa2 = F.softplus(self.predictor_kappa(z2)) + 1e-6
        kappa1 = kappa1.squeeze(-1)  # Converts shape from [128, 1] to [128]
        kappa2 = kappa2.squeeze(-1)
        dist1 = PowerSpherical(mu1, kappa1)
        dist2 = PowerSpherical(mu2, kappa2)
        w1 = dist1.rsample() # shape: [batch_size, d]
        w2 = dist2.rsample()
        p1 = self.decoder(w1)
        p2 = self.decoder(w2)
        return z1, z2, dist1, dist2, p1, p2


def loss_fn(p1, p2, z1, z2):
    def D(p, z, detach=True):
        if detach:
            z = z.detach()
        p = F.normalize(p, dim=-1)
        z = F.normalize(z, dim=-1)
        return -(p * z).sum(dim=-1).mean()

    return 0.5 * (D(p1, z2) + D(p2, z1))


def loss_fn1(p1, p2, z1, z2, dist1, dist2, alpha=1, beta=.00001):
    def kl_divergence(dist):
        """
        Compute KL(PowerSpherical(mu, kappa) || Uniform(S^{d-1}))
        using only the distribution object.

        Args:
            dist: a PowerSpherical distribution (from `power_spherical`)
                  where `dist.loc` is [batch_size, d]
                  and `dist.scale` is [batch_size] or [batch_size, 1]

        Returns:
            Tensor of shape [batch_size], representing KL per sample
        """
        kappa = dist.scale.squeeze(-1) if dist.scale.ndim == 2 else dist.scale
        d = dist.loc.size(-1)

        eps = 1e-8
        kappa = kappa.clamp(min=eps)

        term1 = torch.log(kappa)
        term2 = -(d / 2) * math.log(2 * math.pi)
        term3 = -torch.log(1 - torch.exp(-kappa) + eps)
        term4 = math.log(2) + (d / 2) * math.log(math.pi) - torch.lgamma(torch.tensor(d / 2., device=kappa.device))
        kl = term1 + term2 + term3 + term4
        return kl.mean()

    def D(p, z, detach=True):
        if detach:
            z = z.detach()
        p = F.normalize(p, dim=-1)
        z = F.normalize(z, dim=-1)
        return -(p * z).sum(dim=-1).mean()

    loss_similarity = 0.5* (D(p1, z2) + D(p2, z1))
    loss_recon = 0 # D(p1, z1) + D(p2, z2)
    loss_kl = 0 # kl_divergence(dist1) + kl_divergence(dist2)
    total_loss = loss_similarity + alpha * loss_recon + beta * loss_kl
    return total_loss, loss_similarity, loss_recon, loss_kl


def train(model, dataloader, learning_rate, device, epochs=100, save_interval=100, save_prefix='SimSiamVAE'):
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler(device='cuda')
    model.train()
    loss_history = []

    for epoch in range(epochs):
        train_loss = 0
        for batch_idx, (x1, x2) in enumerate(dataloader):
            x1, x2 = x1.to(device), x2.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast(device_type='cuda', enabled=True):
                z1, z2, dist1, dist2, p1, p2 = model(x1, x2)
                loss = loss_fn(p1, p2, z1, z2)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

        avg_loss = train_loss / len(dataloader)
        loss_history.append(avg_loss)
        print(f'====> Epoch [{epoch + 1}/{epochs}], Avg Total Loss: {avg_loss:.4f}')
        scheduler.step()

        if (epoch + 1) % save_interval == 0:
            save_path = f'{save_prefix}_{epoch + 1}.pth'
            torch.save(model.state_dict(), save_path)
            print(f'Model saved to {save_path}')



    final_save_path = f'{save_prefix}_final.pth'
    torch.save(model.state_dict(), final_save_path)
    print(f'Final model saved to {final_save_path}')
    return loss_history


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    path = '../../../MachineLearning/data/'
    train_dataset = FashionMNISTDataset(path=path, train=True)
    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

    # Training
    model = SimSiamVAE(feature_dim=512, latent_dim=64, variable_dim=32, backbone='cnn-mnist').to(device)
    print(model)
    print('Model Size: {}'.format(param_count(model)))
    loss_history = train(model, train_dataloader, learning_rate=0.001, device=device, epochs=200, save_interval=50)
    display_loss(loss_history)
