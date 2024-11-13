import os
import torch
import random
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms, datasets
from PIL import Image, ImageFilter


class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""
    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""
    def __init__(self, sigma=None):
        if sigma is None:
            sigma = [.1, 2.]
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class CIFAR10Dataset(Dataset):
    def __init__(self, path, train=True, augmented=True, model='resnet_torch'):
        self.train = train
        self.augmented = augmented
        self.data = datasets.CIFAR10(root=path, train=train, download=True, transform=None)
        image_size = 224 if model == 'resnet_torch' else 32

        if self.augmented:
            self.transform = TwoCropsTransform(transforms.Compose([
                transforms.RandomResizedCrop(size=image_size, scale=(0.2, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                GaussianBlur(sigma=[0.1, 2.0]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
            ]))
        else:
            self.transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, target = self.data[idx]

        if self.train and self.augmented:
            x1, x2 = self.transform(img)
            return x1, x2
        else:
            img = self.transform(img)
            return img, target


class SVHNDataset(Dataset):
    def __init__(self, path, train=True, augmented=True):
        path = path + 'SVHN'
        self.split = 'train' if train else 'test'
        self.augmented = augmented
        self.data = datasets.SVHN(root=path, split=self.split, download=True, transform=None)

        if self.augmented:
            self.transform = TwoCropsTransform(transforms.Compose([
                transforms.RandomResizedCrop(size=32, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                GaussianBlur(sigma=[0.1, 2.0]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4377, 0.4438, 0.4728], std=[0.1980, 0.2010, 0.1970])
            ]))
        else:
            self.transform = transforms.Compose([
                transforms.Resize(32),
                transforms.CenterCrop(32),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4377, 0.4438, 0.4728], std=[0.1980, 0.2010, 0.1970])
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, target = self.data[idx]

        if self.split == 'train' and self.augmented:
            x1, x2 = self.transform(img)
            return x1, x2
        else:
            img = self.transform(img)
            return img, target


class FashionMNISTDataset(Dataset):
    def __init__(self, path, train=True, augmented=True):
        self.train = train
        self.augmented = augmented
        self.data = datasets.FashionMNIST(root=path, train=train, download=True, transform=None)

        if self.augmented:
            self.transform = TwoCropsTransform(transforms.Compose([
                transforms.RandomResizedCrop(size=28, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])  # grayscale
            ]))
        else:
            self.transform = transforms.Compose([
                transforms.Resize(28),
                transforms.CenterCrop(28),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, target = self.data[idx]

        if self.train and self.augmented:
            x1, x2 = self.transform(img)
            return x1, x2
        else:
            img = self.transform(img)
            return img, target


class CIFAR100Dataset(Dataset):
    def __init__(self, path, train=True, augmented=True, model='resnet_torch'):
        self.train = train
        self.augmented = augmented
        self.data = datasets.CIFAR100(root=path, train=train, download=True, transform=None)
        image_size = 224 if model == 'resnet_torch' else 32

        if self.augmented:
            self.transform = TwoCropsTransform(transforms.Compose([
                transforms.RandomResizedCrop(size=image_size, scale=(0.2, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                GaussianBlur(sigma=[0.1, 2.0]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
            ]))
        else:
            self.transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, target = self.data[idx]

        if self.train and self.augmented:
            x1, x2 = self.transform(img)
            return x1, x2
        else:
            img = self.transform(img)
            return img, target


class ImageNet100Dataset(Dataset):
    def __init__(self, path, train=True, augmented=True):
        self.train = train
        self.augmented = augmented
        self.root = path + 'ImageNet100/'
        if train:
            self.subdirs = [f'train.X{i}' for i in range(1, 5)]
        else:
            self.subdirs = ['val.X']

        self.imgs = []
        self.labels = []
        self.class_to_idx = {}
        self.idx_to_class = {}
        class_idx = 0

        for subdir in self.subdirs:
            subdir_path = os.path.join(self.root, subdir)
            if not os.path.exists(subdir_path):
                raise FileNotFoundError(f"The directory {subdir_path} does not exist.")

            classes = [d.name for d in os.scandir(subdir_path) if d.is_dir()]
            for cls in classes:
                if cls not in self.class_to_idx:
                    self.class_to_idx[cls] = class_idx
                    self.idx_to_class[class_idx] = cls
                    class_idx += 1

                cls_path = os.path.join(subdir_path, cls)
                self.imgs += [os.path.join(cls_path, img) for img in os.listdir(cls_path) if img.endswith('.JPEG')]
                self.labels += [self.class_to_idx[cls]] * len(os.listdir(cls_path))

        if self.augmented:
            self.transform = TwoCropsTransform(transforms.Compose([
                transforms.RandomResizedCrop(size=224, scale=(0.2, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]))
        else:
            self.transform = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        label = self.labels[idx]
        img = Image.open(img_path).convert('RGB')

        if self.train and self.augmented:
            x1, x2 = self.transform(img)
            return x1, x2
        else:
            img = self.transform(img)
            return img, label


class DsrpitesDataset(Dataset):
    def __init__(self, npy_files):
        self.data = np.concatenate([np.load(file) for file in npy_files], axis=0)
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample


class Shapes3DDataset(Dataset):
    def __init__(self, npy_files):
        self.data = np.concatenate([np.load(file) for file in npy_files], axis=0)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.data[idx]
        sample = Image.fromarray(sample)
        if self.transform:
            sample = self.transform(sample)
        return sample


if __name__ == "__main__":
    SVHN = SVHNDataset(path='../data/', train=True)
    FashionMNIST = FashionMNISTDataset(path='../data/', train=True)
    CIFAR10 = CIFAR10Dataset(path='../data/', train=True)
    ImageNet = ImageNet100Dataset(path='../data/', train=True)
    print(1)