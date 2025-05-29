import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from base.utils import param_count, display_loss
from base.base_model import Encoder, Projector
from base.dataloader import FashionMNISTDataset
from SimSiam import Predictor
from torch.distributions import constraints
from torch.distributions.distribution import Distribution


class PowerSpherical(Distribution):
    arg_constraints = {'loc': constraints.real, 'scale': constraints.positive}
    support = constraints.real

    def __init__(self, loc, scale, validate_args=None):
        self.loc = loc
        self.scale = scale
        batch_shape, event_shape = loc.shape[:-1], loc.shape[-1:]
        super(PowerSpherical, self).__init__(batch_shape, event_shape, validate_args)

    def log_prob(self, value):
        # Normalize loc and value (optional but safer if you expect unit vectors)
        value = F.normalize(value, dim=-1)

        # Clamp scale to avoid log(0) or division by 0
        scale = self.scale.clamp(min=1e-3, max=10.0)

        # Compute diff and safe log-prob per sample
        diff = value - self.loc
        squared_term = -0.5 * ((diff / scale) ** 2).sum(dim=-1)  # shape: [batch]
        log_term = -scale.log().squeeze(-1)  # shape: [batch] if scale is [batch, 1]

        return squared_term + log_term  # shape: [batch]

    @property
    def mean(self):
        return self.loc

    @property
    def variance(self):
        return self.scale ** 2


class PredictorKappa(nn.Module):
    def __init__(self, latent_dim):
        super(PredictorKappa, self).__init__()
        self.fc1 = nn.Linear(latent_dim, latent_dim // 4)
        self.bn1 = nn.BatchNorm1d(latent_dim // 4)
        self.fc2 = nn.Linear(latent_dim // 4, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.fc2(x)
        return x


class SimSiamVI(nn.Module):
    def __init__(self, feature_dim, latent_dim, backbone='resnet-torch', freeze_backbone=False):
        super(SimSiamVI, self).__init__()
        self.encoder = Encoder(backbone=backbone, freeze_backbone=freeze_backbone)
        self.projector = Projector(feature_dim=feature_dim, latent_dim=latent_dim)
        self.predictor_mu = Predictor(latent_dim=latent_dim)
        self.predictor_kappa = PredictorKappa(latent_dim=feature_dim)

    def forward(self, x1, x2):
        y1 = self.encoder(x1)
        y2 = self.encoder(x2)

        z1 = self.projector(y1)
        z2 = self.projector(y2)

        mu1 = F.normalize(self.predictor_mu(z1), dim=-1)  # Unit norm
        mu2 = F.normalize(self.predictor_mu(z2), dim=-1)

        kappa1 = F.softplus(self.predictor_kappa(y1)) + 1e-3  # Ensure kappa is positive
        kappa2 = F.softplus(self.predictor_kappa(y2)) + 1e-3

        dist1 = PowerSpherical(mu1, kappa1)
        dist2 = PowerSpherical(mu2, kappa2)

        return z1, z2, dist1, dist2


def loss_fn(z1, z2, dist1, dist2):
    z1 = z1.detach()
    z2 = z2.detach()
    loss = -dist1.log_prob(z2).mean() - dist2.log_prob(z1).mean()
    return loss


def train(model, dataloader, learning_rate, device, epochs=100, save_interval=100, save_prefix='SimSiamVI_CIFAR'):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
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
                z1, z2, dist1, dist2 = model(x1, x2)
                loss = loss_fn(z1, z2, dist1, dist2)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

        avg_loss = train_loss / len(dataloader)
        loss_history.append(avg_loss)
        print(f'====> Epoch [{epoch+1}/{epochs}], Avg Total Loss: {avg_loss:.4f}')
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
    path = '../../MachineLearning/data/'
    train_dataset = FashionMNISTDataset(path=path, train=True)
    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

    # Training
    model = SimSiamVI(feature_dim=512, latent_dim=64, backbone='cnn-mnist').to(device)
    print(model)
    print('Model Size: {}'.format(param_count(model)))
    loss_history = train(model, train_dataloader, learning_rate=0.001, device=device, epochs=200, save_interval=50, save_prefix='SimSiamVI_CIFAR')
    display_loss(loss_history)
