import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from base.utils import param_count, display_loss
from base.base_model import Encoder, Projector
from base.dataloader import FashionMNISTDataset


class SimCLR(nn.Module):
    def __init__(self, feature_dim, latent_dim, backbone='resnet-torch', freeze_backbone=False):
        super(SimCLR, self).__init__()
        self.encoder = Encoder(backbone=backbone, freeze_backbone=freeze_backbone)
        self.projector = Projector(feature_dim=feature_dim, latent_dim=latent_dim)

    def forward(self, x):
        y = self.encoder(x)
        z = self.projector(y)
        z = F.normalize(z, dim=-1)
        return z


def loss_fn(z_i, z_j, temperature=0.5):
    batch_size = z_i.shape[0]

    representations = torch.cat([z_i, z_j], dim=0)
    similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)

    sim_ij = torch.diag(similarity_matrix, batch_size)
    sim_ji = torch.diag(similarity_matrix, -batch_size)

    positives = torch.cat([sim_ij, sim_ji], dim=0)
    nominator = torch.exp(positives / temperature)

    negatives_mask = (torch.ones_like(similarity_matrix) - torch.eye(2 * batch_size, device=similarity_matrix.device)).bool()
    denominator = torch.exp(similarity_matrix / temperature)[negatives_mask].view(2 * batch_size, -1).sum(dim=1)

    loss = -torch.log(nominator / denominator).mean()
    return loss


def train(model, dataloader, learning_rate, device, epochs=100, save_interval=100, save_prefix='SimCLR_CIFAR'):
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
            z1 = model(x1)
            z2 = model(x2)
            loss = loss_fn(z1, z2)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

        avg_loss = train_loss / D
        loss_history.append(avg_loss)
        print(f'====> Epoch [{epoch + 1}/{epochs}], Avg Loss: {avg_loss:.4f}')

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
    train_dataset = FashionMNISTDataset(path='../data/', train=True)
    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

    # Training
    model = SimCLR(feature_dim=512, latent_dim=64, backbone='cnn-mnist').to(device)
    print(model)
    print('Model Size: {}'.format(param_count(model)))
    loss_history = train(model, train_dataloader, learning_rate=0.001, device=device, epochs=200, save_interval=50, save_prefix='SimCLR_CIFAR')
    display_loss(loss_history)
