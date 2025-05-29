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

#-----------------------------

class CIFAR10Dataset(Dataset):
    def __init__(self, path, train=True, augmented=True, model='resnet_cifar'):
        self.train = train
        self.augmented = augmented
        self.data = datasets.CIFAR10(root=path, train=train, download=True, transform=None)
        image_size = 32 if model == 'resnet_cifar' else 224

        if self.augmented:
            self.transform = TwoCropsTransform(transforms.Compose([
                transforms.RandomResizedCrop(size=image_size, scale=(0.2, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                     std=[0.2023, 0.1994, 0.2010])
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


class CIFAR100Dataset(Dataset):
    def __init__(self, path, train=True, augmented=True, model='resnet_cifar'):
        self.train = train
        self.augmented = augmented
        self.data = datasets.CIFAR100(root=path, train=train, download=True, transform=None)
        image_size = 32 if model == 'resnet_cifar' else 224

        if self.augmented:
            self.transform = TwoCropsTransform(transforms.Compose([
                transforms.RandomResizedCrop(size=image_size, scale=(0.2, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                     std=[0.2675, 0.2565, 0.2761])
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

# -----------------------------

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

#-----------------------------

class ImageNet100Dataset(Dataset):
    def __init__(self, path, train=True, augmented=True):
        self.train = train
        self.augmented = augmented
        self.root = os.path.join(path, 'ImageNet100/')
        self.subdirs = [f'train.X{i}' for i in range(1, 5)] if train else ['val.X']
        self.imgs = []
        self.labels = []
        self.class_to_idx = {}
        self.idx_to_class = {}
        class_idx = 0

        for subdir in self.subdirs:
            subdir_path = os.path.join(self.root, subdir)
            if not os.path.exists(subdir_path):
                raise FileNotFoundError(f"{subdir_path} does not exist")

            for cls in sorted(os.listdir(subdir_path)):
                cls_path = os.path.join(subdir_path, cls)
                if not os.path.isdir(cls_path):
                    continue

                if cls not in self.class_to_idx:
                    self.class_to_idx[cls] = class_idx
                    self.idx_to_class[class_idx] = cls
                    class_idx += 1

                image_files = [
                    os.path.join(cls_path, img)
                    for img in os.listdir(cls_path)
                    if img.lower().endswith(('.jpeg', '.jpg'))
                ]
                self.imgs.extend(image_files)
                self.labels.extend([self.class_to_idx[cls]] * len(image_files))

        if self.augmented:
            self.transform = TwoCropsTransform(transforms.Compose([
                transforms.Resize(256),
                transforms.RandomResizedCrop(224, scale=(0.2, 1.0), interpolation=Image.BICUBIC),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                GaussianBlur(sigma=[0.1, 2.0]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ]))

        else:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        path = self.imgs[idx]
        label = self.labels[idx]
        img = Image.open(path).convert('RGB')

        if self.train and self.augmented:
            x1, x2 = self.transform(img)
            return x1, x2
        else:
            img = self.transform(img)
            return img, label


class ImageNet100HFDataset(Dataset):
    def __init__(self, hf_dataset, train=True, augmented=True):
        self.train = train
        self.augmented = augmented
        self.dataset = hf_dataset  # Hugging Face dataset object

        if self.augmented:
            self.transform = TwoCropsTransform(transforms.Compose([
                transforms.Resize(256),
                transforms.RandomResizedCrop(224, scale=(0.2, 1.0), interpolation=Image.BICUBIC),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                GaussianBlur(sigma=[0.1, 2.0]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ]))
        else:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = self.dataset[idx]['image']

        # Ensure PIL Image and convert to RGB
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        img = img.convert('RGB')

        if self.train and self.augmented:
            x1, x2 = self.transform(img)
            return x1, x2
        else:
            img = self.transform(img)
            return img, self.dataset[idx]['label']

#-----------------------------

class TinyImageNet200Dataset(Dataset):
    def __init__(self, path, train=True, augmented=True):
        self.train = train
        self.augmented = augmented
        self.root = os.path.join(path, 'ImageNet200tiny')
        self.imgs = []
        self.labels = []
        self.class_to_idx = {}
        self.idx_to_class = {}
        class_idx = 0

        if self.train:
            train_dir = os.path.join(self.root, 'train')
            for cls in sorted(os.listdir(train_dir)):
                cls_path = os.path.join(train_dir, cls, 'images')
                if not os.path.isdir(cls_path):
                    continue

                if cls not in self.class_to_idx:
                    self.class_to_idx[cls] = class_idx
                    self.idx_to_class[class_idx] = cls
                    class_idx += 1

                image_files = [
                    os.path.join(cls_path, img)
                    for img in os.listdir(cls_path)
                    if img.lower().endswith(('.jpeg', '.jpg', '.png'))
                ]
                self.imgs.extend(image_files)
                self.labels.extend([self.class_to_idx[cls]] * len(image_files))

        else:
            val_dir = os.path.join(self.root, 'val')
            val_img_dir = os.path.join(val_dir, 'images')
            val_annotations_path = os.path.join(val_dir, 'val_annotations.txt')

            # Parse validation annotations
            val_map = {}
            with open(val_annotations_path, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    val_map[parts[0]] = parts[1]

            for img_file, cls in val_map.items():
                img_path = os.path.join(val_img_dir, img_file)

                if cls not in self.class_to_idx:
                    self.class_to_idx[cls] = class_idx
                    self.idx_to_class[class_idx] = cls
                    class_idx += 1

                self.imgs.append(img_path)
                self.labels.append(self.class_to_idx[cls])

        if self.augmented:
            self.transform = TwoCropsTransform(transforms.Compose([
                transforms.Resize(256),
                transforms.RandomResizedCrop(224, scale=(0.2, 1.0), interpolation=Image.BICUBIC),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ]))
        else:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        path = self.imgs[idx]
        label = self.labels[idx]
        img = Image.open(path).convert('RGB')

        if self.train and self.augmented:
            x1, x2 = self.transform(img)
            return x1, x2
        else:
            img = self.transform(img)
            return img, label

#-----------------------------

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
    from datasets import load_dataset

    full_dataset = load_dataset("clane9/imagenet-100")
    train_hf = full_dataset['train']
    test_hf = full_dataset['validation']

    # Create your datasets
    train_dataset = ImageNet100HFDataset(train_hf, train=True, augmented=True)
    eval_dataset_train = ImageNet100HFDataset(train_hf, train=False, augmented=False)
    eval_dataset_test = ImageNet100HFDataset(test_hf, train=False, augmented=False)
    print(1)
