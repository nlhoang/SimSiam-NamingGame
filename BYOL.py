import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from copy import deepcopy
from base.utils import param_count, display_loss
from base.base_model import Encoder, Projector
from base.dataloader import FashionMNISTDataset
from SimSiam import Predictor


class BYOL(nn.Module):
    def __init__(self, feature_dim, latent_dim, backbone='resnet-torch', freeze_backbone=False, momentum=0.996):
        super(BYOL, self).__init__()
        # Online network
        self.encoder_online = Encoder(backbone=backbone, freeze_backbone=freeze_backbone)
        self.projector_online = Projector(feature_dim=feature_dim, latent_dim=latent_dim)
        self.predictor = Predictor(latent_dim=latent_dim)

        # Target network (EMA of online)
        self.encoder_target = deepcopy(self.encoder_online)
        self.projector_target = deepcopy(self.projector_online)

        for param in self.encoder_target.parameters():
            param.requires_grad = False
        for param in self.projector_target.parameters():
            param.requires_grad = False

        self.momentum = momentum

    @torch.no_grad()
    def update_target_network(self):
        for online, target in zip(self.encoder_online.parameters(), self.encoder_target.parameters()):
            target.data = self.momentum * target.data + (1 - self.momentum) * online.data
        for online, target in zip(self.projector_online.parameters(), self.projector_target.parameters()):
            target.data = self.momentum * target.data + (1 - self.momentum) * online.data

    def forward(self, x1, x2):
        # Online network
        y1 = self.encoder_online(x1)
        y2 = self.encoder_online(x2)
        z1 = self.projector_online(y1)
        z2 = self.projector_online(y2)
        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        # Target network (no grad)
        with torch.no_grad():
            y1_t = self.encoder_target(x1)
            y2_t = self.encoder_target(x2)
            z1_t = self.projector_target(y1_t).detach()
            z2_t = self.projector_target(y2_t).detach()

        return p1, p2, z1_t, z2_t


def byol_loss(p1, p2, z1_t, z2_t):
    def loss_fn(p, z):
        p = F.normalize(p, dim=-1)
        z = F.normalize(z, dim=-1)
        return 2 - 2 * (p * z).sum(dim=-1).mean()
    return loss_fn(p1, z2_t) + loss_fn(p2, z1_t)


def train(model, dataloader, learning_rate, device, epochs=100, save_interval=100, save_prefix='BYOL_CIFAR'):
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
                p1, p2, z1_t, z2_t = model(x1, x2)
                loss = byol_loss(p1, p2, z1_t, z2_t)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            model.update_target_network()
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
    path = '../../MachineLearning/data/'
    train_dataset = FashionMNISTDataset(path=path, train=True)
    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)

    model = BYOL(feature_dim=512, latent_dim=64, backbone='cnn-mnist').to(device)
    print(model)
    print('Model Size: {}'.format(param_count(model)))
    loss_history = train(model, train_dataloader, learning_rate=0.05, device=device, epochs=200, save_interval=50, save_prefix='BYOL_CIFAR')
    display_loss(loss_history)


