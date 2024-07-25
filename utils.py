from torch.utils.data import Dataset
from torchvision import transforms, datasets
import matplotlib.pyplot as plt


class CIFAR10Dataset(Dataset):
    def __init__(self, train=True):
        self.data = datasets.CIFAR10(root='../data', train=train, download=True, transform=None)
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(size=32, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, _ = self.data[idx]
        x1 = self.transform(img)
        x2 = self.transform(img)
        return x1, x2


def param_count(model):
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return num_params


def display_loss(loss):
    plt.plot(loss)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.show()
