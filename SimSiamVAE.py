import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils import param_count, display_loss
from base_model import Encoder, Projector
from dataloader import FashionMNISTDataset


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
        self.fc1 = nn.Linear(variable_dim, latent_dim // 4)
        self.bn1 = nn.BatchNorm1d(latent_dim // 4)
        self.fc2 = nn.Linear(latent_dim // 4, latent_dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = torch.sigmoid(self.fc2(x))
        return x


class SimSiamVAE(nn.Module):
    def __init__(self, feature_dim, latent_dim, variable_dim, backbone='resnet_torch', freeze_backbone=False):
        super(SimSiamVAE, self).__init__()
        self.encoder = Encoder(backbone=backbone, freeze_backbone=freeze_backbone)
        self.projector = Projector(feature_dim=feature_dim, latent_dim=latent_dim)
        self.predictor_mu = PredictorEncoder(latent_dim=latent_dim, variable_dim=variable_dim)
        self.predictor_logvar = PredictorEncoder(latent_dim=latent_dim, variable_dim=variable_dim)
        self.decoder = PredictorDecoder(variable_dim=variable_dim, latent_dim=latent_dim)

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


def loss_fn(p1, p2, z1, z2, mu1, logvar1, mu2, logvar2, loss_type='cosine', beta=0.0001):
    def kl_divergence(mu, logvar):
        return 0.5 * torch.sum(mu ** 2 + logvar.exp() - logvar - 1, dim=-1).mean()

    def mse_loss(x, y):
        return F.mse_loss(x, y)

    def negative_cosine_similarity(x, y):
        x = F.normalize(x, dim=-1)
        y = F.normalize(y, dim=-1)
        return - (x * y).sum(dim=-1).mean()

    z1 = z1.detach()
    z2 = z2.detach()

    if loss_type == 'cosine':
        loss_similarity = negative_cosine_similarity(p1, z2) + negative_cosine_similarity(p2, z1)
    else:  # loss_type == 'mse':
        loss_similarity = mse_loss(p1, z2) + mse_loss(p2, z1)

    loss_kl = kl_divergence(mu1, logvar1) + kl_divergence(mu2, logvar2)
    total_loss = loss_similarity + beta * loss_kl
    return total_loss, loss_similarity, beta * loss_kl


def train(model, dataloader, learning_rate, device, epochs=100, save_interval=100, save_prefix='SimSiamVAE'):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    model.train()
    D = len(dataloader.dataset)
    loss_history = []

    for epoch in range(epochs):
        train_loss = 0
        similarity_loss = 0
        kl_loss = 0
        for batch_idx, (x1, x2) in enumerate(dataloader):
            x1, x2 = x1.to(device), x2.to(device)
            optimizer.zero_grad()
            z1, z2, mu1, logvar1, mu2, logvar2, p1, p2 = model(x1, x2)
            loss, loss_similarity, loss_kl = loss_fn(p1, p2, z1, z2, mu1, logvar1, mu2, logvar2)
            train_loss += loss.item()
            similarity_loss += loss_similarity.item()
            kl_loss += loss_kl.item()
            loss.backward()
            optimizer.step()

        avg_loss = train_loss / D
        avg_similarity_loss = similarity_loss / D
        avg_kl_loss = kl_loss / D
        loss_history.append(avg_loss)
        print(f'====> Epoch [{epoch + 1}/{epochs}], Avg Loss: {avg_loss:.4f}, Similarity Loss: {avg_similarity_loss:.4f}, KL Loss: {avg_kl_loss:.4f}')

        if (epoch + 1) % save_interval == 0:
            save_path = f'{save_prefix}_{epoch + 1}.pth'
            torch.save(model.state_dict(), save_path)
            print(f'Model saved to {save_path}')

        scheduler.step()

    final_save_path = f'{save_prefix}_final.pth'
    torch.save(model.state_dict(), final_save_path)
    print(f'Final model saved to {final_save_path}')
    return loss_history


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
    print(f"Using device: {device}")
    train_dataset = FashionMNISTDataset(path='../data/', train=True)
    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

    # Training
    model = SimSiamVAE(feature_dim=512, latent_dim=64, variable_dim=32, backbone='cnn-mnist').to(device)
    print(model)
    print('Model Size: {}'.format(param_count(model)))
    loss_history = train(model, train_dataloader, learning_rate=0.001, device=device, epochs=200, save_interval=50)
    display_loss(loss_history)
