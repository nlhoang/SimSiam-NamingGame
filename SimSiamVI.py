import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.utils.data import DataLoader
from utils import CIFAR10Dataset, display_loss


class PowerSpherical(Distribution):
    arg_constraints = {'loc': constraints.real, 'scale': constraints.positive}
    support = constraints.real

    def __init__(self, loc, scale, validate_args=None):
        self.loc = loc
        self.scale = scale
        batch_shape, event_shape = loc.shape[:-1], loc.shape[-1:]
        super(PowerSpherical, self).__init__(batch_shape, event_shape, validate_args)

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        norm = torch.randn(shape, dtype=self.loc.dtype, device=self.loc.device)
        norm = norm / norm.norm(dim=-1, keepdim=True)
        scaled = norm * (1 - torch.exp(-self.scale))
        return self.loc + scaled

    def log_prob(self, value):
        diff = value - self.loc
        return -0.5 * ((diff / self.scale) ** 2).sum(dim=-1) - self.scale.log().sum()

    @property
    def mean(self):
        return self.loc

    @property
    def variance(self):
        return self.scale ** 2


class Encoder(nn.Module):
    def __init__(self, feature_dim):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.fc_latent = nn.Linear(256 * 4 * 4, feature_dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.adaptive_avg_pool2d(x, (4, 4))
        x = x.view(x.size(0), -1)  # Flatten the features
        latent = self.fc_latent(x)
        return latent


class Projector(nn.Module):
    def __init__(self, feature_dim, latent_dim):
        super(Projector, self).__init__()
        self.fc1 = nn.Linear(feature_dim, latent_dim)
        self.bn1 = nn.BatchNorm1d(latent_dim)
        self.fc2 = nn.Linear(latent_dim, latent_dim)
        self.bn2 = nn.BatchNorm1d(latent_dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.bn2(self.fc2(x))
        return x


class Predictor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Predictor, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.bn1 = nn.BatchNorm1d(input_dim)
        self.fc2 = nn.Linear(input_dim, output_dim)
        self.bn2 = nn.BatchNorm1d(output_dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.bn2(self.fc2(x))
        return x


class SimSiamVI(nn.Module):
    def __init__(self, feature_dim, latent_dim):
        super(SimSiamVI, self).__init__()
        self.encoder = Encoder(feature_dim)
        self.projector = Projector(feature_dim, latent_dim)
        self.predictor_mu = Predictor(latent_dim, latent_dim)
        self.predictor_kappa = Predictor(feature_dim, latent_dim)

    def forward(self, x1, x2):
        y1 = self.encoder(x1)
        y2 = self.encoder(x2)

        z1 = self.projector(y1)
        z2 = self.projector(y2)

        mu1 = self.predictor_mu(z1)
        mu2 = self.predictor_mu(z2)

        kappa1 = F.softplus(self.predictor_kappa(y1)) + 1e-6  # Ensure kappa is positive
        kappa2 = F.softplus(self.predictor_kappa(y2)) + 1e-6  # Ensure kappa is positive

        dist1 = PowerSpherical(mu1, kappa1)
        dist2 = PowerSpherical(mu2, kappa2)

        return z1, z2, dist1, dist2


def loss_fn(z1, z2, dist1, dist2):
    z1 = z1.detach()
    z2 = z2.detach()
    loss = -dist1.log_prob(z2).mean() - dist2.log_prob(z1).mean()
    return loss


def train(model, dataloader, learning_rate, device, epochs=100, saved='VI_SimSiam_cifar10.pth'):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    model.train()
    D = len(dataloader.dataset)
    loss_history = []

    for epoch in range(epochs):
        train_loss = 0
        for batch_idx, (x1, x2) in enumerate(dataloader):
            x1, x2 = x1.to(device), x2.to(device)
            optimizer.zero_grad()
            z1, z2, dist1, dist2 = model(x1, x2)
            loss = loss_fn(z1, z2, dist1, dist2)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

        avg_loss = train_loss / D
        loss_history.append(avg_loss)
        print(f'====> Epoch: {epoch}, Avg Total Loss: {avg_loss:.4f}')
        scheduler.step()

    torch.save(model.state_dict(), saved)
    return loss_history


train_dataset = CIFAR10Dataset(train=True)
train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)

# Training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimSiamVI(feature_dim=256, latent_dim=128).to(device)
loss_history = train(model, train_dataloader, learning_rate=0.001, device=device, epochs=100)
display_loss(loss_history)
