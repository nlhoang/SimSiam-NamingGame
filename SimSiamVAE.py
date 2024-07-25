import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils import CIFAR10Dataset, display_loss


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


class Decoder(nn.Module):
    def __init__(self, variable_dim, latent_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(variable_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, latent_dim)
        self.bn3 = nn.BatchNorm1d(latent_dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = torch.sigmoid(self.fc3(x))
        return x


class SimSiamVAE(nn.Module):
    def __init__(self, feature_dim, latent_dim, variable_dim):
        super(SimSiamVAE, self).__init__()
        self.encoder = Encoder(feature_dim)
        self.projector = Projector(feature_dim, latent_dim)
        self.predictor_mu = Predictor(latent_dim, variable_dim)
        self.predictor_logvar = Predictor(latent_dim, variable_dim)
        self.decoder = Decoder(variable_dim, latent_dim)

    def forward(self, x1, x2):
        y1 = self.encoder(x1)
        y2 = self.encoder(x2)

        z1 = self.projector(y1)
        z2 = self.projector(y2)

        mu1 = self.predictor_mu(z1)
        mu2 = self.predictor_mu(z2)

        logvar1 = self.predictor_logvar(z1)
        logvar2 = self.predictor_logvar(z2)

        w1 = self.reparameterize(mu1, logvar1)
        w2 = self.reparameterize(mu2, logvar2)

        p1 = self.decoder(w1)
        p2 = self.decoder(w2)

        return z1, z2, mu1, logvar1, mu2, logvar2, p1, p2

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = torch.empty_like(std).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu


def negative_cosine_similarity(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return - (x * y).sum(dim=-1).mean()


def mse_loss(x, y):
    return F.mse_loss(x, y)


def kl_divergence(mu, logvar):
    return 0.5 * torch.sum(mu**2 + logvar.exp() - logvar - 1, dim=-1).mean()


def loss_fn(p1, p2, z1, z2, mu1, logvar1, mu2, logvar2, loss_type='cosine', loss_kl=False):
    z1 = z1.detach()
    z2 = z2.detach()

    if loss_type == 'cosine':
        loss_recon = negative_cosine_similarity(p1, z1) + negative_cosine_similarity(p2, z2)
        loss_simsiam = negative_cosine_similarity(p1, z2) + negative_cosine_similarity(p2, z1)
    elif loss_type == 'mse':
        loss_recon = mse_loss(p1, z1) + mse_loss(p2, z2)
        loss_simsiam = mse_loss(p1, z2) + mse_loss(p2, z1)
    else:
        raise ValueError(f"Invalid loss_type: {loss_type}")

    if loss_kl:
        loss_kl1 = kl_divergence(mu1, logvar1)
        loss_kl2 = kl_divergence(mu2, logvar2)
    else:
        loss_kl1 = 0
        loss_kl2 = 0

    total_loss = loss_recon + loss_simsiam + loss_kl1 + loss_kl2
    return total_loss, loss_recon, loss_simsiam, loss_kl1, loss_kl2


def train(model, dataloader, learning_rate, device, epochs=100, saved='SimSiamVAE_cifar10.pth'):
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
            z1, z2, mu1, logvar1, mu2, logvar2, p1, p2 = model(x1, x2)
            loss, loss_recon, loss_simsiam, loss_kl1, loss_kl2 = loss_fn(p1, p2, z1, z2, mu1, logvar1, mu2, logvar2)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

        avg_loss = train_loss / D
        loss_history.append(avg_loss)
        print(f'====> Epoch: {epoch}, Avg Loss: {avg_loss:.4f}')
        scheduler.step()

    torch.save(model.state_dict(), saved)
    return loss_history


train_dataset = CIFAR10Dataset(train=True)
train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)

# Training
device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
model = SimSiamVAE(feature_dim=256, latent_dim=128, variable_dim=64).to(device)
loss_history = train(model, train_dataloader, learning_rate=0.001, device=device, epochs=100,
                     saved='SimSiamVAE_GS_CIFAR_100.pth')
display_loss(loss_history)

