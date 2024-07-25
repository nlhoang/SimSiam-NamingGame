import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils import param_count, CIFAR10Dataset, display_loss


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
    def __init__(self, latent_dim):
        super(Predictor, self).__init__()
        self.fc1 = nn.Linear(latent_dim, latent_dim)
        self.bn1 = nn.BatchNorm1d(latent_dim)
        self.fc2 = nn.Linear(latent_dim, latent_dim)
        self.bn2 = nn.BatchNorm1d(latent_dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.bn2(self.fc2(x))
        return x


class SimSiam(nn.Module):
    def __init__(self, feature_dim, latent_dim):
        super(SimSiam, self).__init__()
        self.encoder = Encoder(feature_dim)
        self.projector = Projector(feature_dim, latent_dim)
        self.predictor = Predictor(latent_dim)

    def forward(self, x1, x2):
        z1 = self.encoder(x1)
        z2 = self.encoder(x2)

        proj_z1 = self.projector(z1)
        proj_z2 = self.projector(z2)

        p1 = self.predictor(proj_z1)
        p2 = self.predictor(proj_z2)

        proj_z1 = proj_z1.detach()
        proj_z2 = proj_z2.detach()

        return p1, p2, proj_z1, proj_z2


def loss_fn(p1, p2, z1, z2):
    def D(p, z):
        z = z.detach()
        p = F.normalize(p, dim=-1)
        z = F.normalize(z, dim=-1)
        return -(p * z).sum(dim=-1).mean()

    return 0.5 * (D(p1, z2) + D(p2, z1))


def train(model, dataloader, learning_rate, device, epochs=100, saved='SimSiam_cifar10.pth'):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    model.train()
    D = len(dataloader.dataset)
    loss_history = []

    for epoch in range(epochs):
        train_loss = 0
        for batch_idx, (x1, x2) in enumerate(dataloader):
            x1, x2 = x1.to(device), x2.to(device)
            optimizer.zero_grad()
            p1, p2, z1, z2 = model(x1, x2)
            loss = loss_fn(p1, p2, z1, z2)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print(f'Batch {batch_idx}, Loss: {loss.item():.4f}')

        avg_loss = train_loss / D
        loss_history.append(avg_loss)
        print('====> Epoch: {}, Loss: {:.4f}'.format(epoch, avg_loss))
        scheduler.step()

    torch.save(model.state_dict(), saved)
    return loss_history


train_dataset = CIFAR10Dataset(train=True)
train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)

# Training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimSiam(feature_dim=256, latent_dim=128).to(device)
loss_history = train(model, train_dataloader, learning_rate=0.001, device=device, epochs=100)
display_loss(loss_history)
