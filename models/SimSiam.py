import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import param_count, CIFAR10Dataset, display_loss
from models.base_model import Encoder, Projector


class Predictor(nn.Module):
    def __init__(self, latent_dim):
        super(Predictor, self).__init__()
        self.fc1 = nn.Linear(latent_dim, latent_dim // 2)
        self.bn1 = nn.BatchNorm1d(latent_dim // 2)
        self.fc2 = nn.Linear(latent_dim // 2, latent_dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.fc2(x)
        return x


class SimSiam(nn.Module):
    def __init__(self, feature_dim, latent_dim, backbone='resnet_torch'):
        super(SimSiam, self).__init__()
        self.encoder = Encoder(backbone=backbone)
        self.projector = Projector(feature_dim=feature_dim, latent_dim=latent_dim)
        self.predictor = Predictor(latent_dim=latent_dim)

    def forward(self, x1, x2):
        y1 = self.encoder(x1)
        y2 = self.encoder(x2)

        z1 = self.projector(y1)
        z2 = self.projector(y2)

        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        return p1, p2, z1, z2


def loss_fn(p1, p2, z1, z2):
    def D(p, z):
        z = z.detach()
        p = F.normalize(p, dim=-1)
        z = F.normalize(z, dim=-1)
        return -(p * z).sum(dim=-1).mean()

    return 0.5 * (D(p1, z2) + D(p2, z1))


def train(model, dataloader, learning_rate, device, epochs=100, save_interval=100, save_prefix='SimSiam_CIFAR'):
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

        avg_loss = train_loss / D
        loss_history.append(avg_loss)
        print('====> Epoch {}, Loss: {:.4f}'.format(epoch, avg_loss))
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
    print(f"Using device: {device}")
    train_dataset = CIFAR10Dataset(path='../../data/', train=True)
    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

    # Training
    model = SimSiam(feature_dim=512, latent_dim=2048, backbone='cnn').to(device)
    print(model)
    print('Model Size: {}'.format(param_count(model)))
    loss_history = train(model, train_dataloader, learning_rate=0.001, device=device, epochs=200, save_interval=50, save_prefix='SimSiam_CIFAR')
    display_loss(loss_history)
