import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from models import SimSiam, SimSiamVI, SimSiamVAE
from tqdm import tqdm
from utils import CIFAR100Dataset, CIFAR10Dataset, ImageNet100Dataset


def extract_features(model, dataloader, device):
    model.eval()
    features = []
    labels = []

    use_amp = torch.cuda.is_available()
    with torch.no_grad():
        for imgs, targets in tqdm(dataloader, desc="Extracting"):
            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            if use_amp:
                with torch.cuda.amp.autocast():
                    y = model.encoder(imgs)
                    z = model.projector(y)
            else:
                y = model.encoder(imgs)
                z = model.projector(y)

            features.append(z.cpu())
            labels.append(targets.cpu())

    features = torch.cat(features).numpy()
    labels = torch.cat(labels).numpy()
    return features, labels


class LinearClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        x = self.fc(x)
        return x


class MLPClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=512):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x


class DeeperMLPClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=512):
        super(DeeperMLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.fc4(x)
        return x


def train_classifier(classifier, features, labels, num_classes, device, epochs=50):
    if classifier == 'Linear':
        classifier = LinearClassifier(features.shape[1], num_classes).to(device)
    elif classifier == 'MLP':
        classifier = MLPClassifier(features.shape[1], num_classes).to(device)
    else:  # classifier == 'DeeperMLP':
        classifier = DeeperMLPClassifier(features.shape[1], num_classes).to(device)

    optimizer = optim.Adam(classifier.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    dataset = TensorDataset(torch.tensor(features).float(), torch.tensor(labels).long())
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

    classifier.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = classifier(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'====> Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(dataloader):.4f}')
    return classifier


def evaluate_classifier(classifier, features, labels, device, max_k=5):
    dataset = TensorDataset(torch.tensor(features).float(), torch.tensor(labels).long())
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False)

    classifier.eval()
    correct_topk = [0] * max_k  # List to hold correct predictions for each k
    total = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = classifier(inputs)
            _, predicted_topk_all = torch.topk(outputs.data, max_k, dim=1)  # top-k predictions
            total += targets.size(0)
            for k in range(max_k):  # Calculate accuracy for each k
                correct_topk[k] += sum([targets[i] in predicted_topk_all[i, :k+1] for i in range(targets.size(0))])

    accuracy = [100 * correct / total for correct in correct_topk]
    for k in range(max_k):
        print(f'Top-{k+1} Accuracy: {accuracy[k]:.2f}%')
    # return accuracy


model_files_SimSiamVAE_resnet18 = [
    'SimSiamVAE_CIFAR100_50.pth', 'SimSiamVAE_CIFAR100_100.pth', 'SimSiamVAE_CIFAR100_150.pth',
    'SimSiamVAE_CIFAR100_200.pth', 'SimSiamVAE_CIFAR100_250.pth', 'SimSiamVAE_CIFAR100_300.pth',
    'SimSiamVAE_CIFAR100_350.pth', 'SimSiamVAE_CIFAR100_400.pth', 'SimSiamVAE_CIFAR100_450.pth',
    'SimSiamVAE_CIFAR100_500.pth', 'SimSiamVAE_CIFAR100_final.pth']
model_files_SimSiamVAE_CNN = [
    'SimSiamVAE_CIFAR100_100.pth', 'SimSiamVAE_CIFAR100_200.pth', 'SimSiamVAE_CIFAR100_300.pth',
    'SimSiamVAE_CIFAR100_400.pth', 'SimSiamVAE_CIFAR100_500.pth', 'SimSiamVAE_CIFAR100_600.pth',
    'SimSiamVAE_CIFAR100_700.pth', 'SimSiamVAE_CIFAR100_800.pth', 'SimSiamVAE_CIFAR100_900.pth',
    'SimSiamVAE_CIFAR100_1000.pth', 'SimSiamVAE_CIFAR100_final.pth']


def args_define():
    parser = argparse.ArgumentParser(description='Evaluate SimSiamVAE models.')
    parser.add_argument('--eval-model', default='SimSiamVAE', choices=['SimSiam', 'SimSiamVI', 'SimSiamVAE'])
    parser.add_argument('--dataset', default='ImageNet', choices=['CIFAR', 'CIFAR100', 'ImageNet'])
    parser.add_argument('--backbone', default='cnn', choices=['cnn', 'resnet_cifar', 'resnet_torch'])
    parser.add_argument('--num-classes', type=int, default=10, choices=[10, 100], help='number of classes')
    parser.add_argument('--feature-dim', type=int, default=512, help='feature dim from backbone [default: 512]')
    parser.add_argument('--latent-dim', type=int, default=2048, help='latent dim from projector [default: 2048]')
    parser.add_argument('--variable-dim', type=int, default=1024, help='variable dim from predictor [default: 1024]')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs [default: 500]')
    parser.add_argument('--mode', default='single', choices=['single', 'multiple'])
    parser.add_argument('--classifier', default='DeeperMLP', choices=['Linear', 'MLP', 'DeeperMLP'])
    parser.add_argument('--model_list', nargs='+', help='List of model files for multiple evaluation')
    args = parser.parse_args()
    return args


def initialize(args):
    if args.debug:
        args.epochs = 2

    if args.dataset == 'CIFAR':
        args.backbone = 'cnn'
        args.num_classes = 10
        args.feature_dim = 512
        args.latent_dim = 512
        args.variable_dim = 512
    elif args.dataset == 'CIFAR100':
        args.backbone = 'resnet_cifar'
        args.num_classes = 100
        args.feature_dim = 512
        args.latent_dim = 2048
        args.variable_dim = 1024
    else:  # args.dataset == 'ImageNet':
        args.backbone = 'resnet_torch'
        args.num_classes = 100
        args.feature_dim = 512
        args.latent_dim = 2048
        args.variable_dim = 1024


def main():
    args = args_define()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(args)
    IS_SERVER = False
    path = '/data/' if IS_SERVER else '../data/'

    # Load Dataset for feature extraction
    if args.dataset == 'CIFAR':
        train_dataset = CIFAR10Dataset(path=path, train=True, augmented=False)
        test_dataset = CIFAR10Dataset(path=path, train=False, augmented=False)
    elif args.dataset == 'CIFAR100':
        train_dataset = CIFAR100Dataset(path=path, train=True, augmented=False)
        test_dataset = CIFAR100Dataset(path=path, train=False, augmented=False)
    else:   # args.dataset == 'ImageNet':
        train_dataset = ImageNet100Dataset(path=path, train=True, augmented=False)
        test_dataset = ImageNet100Dataset(path=path, train=False, augmented=False)

    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

    if args.mode == 'single':   # Evaluate a single model
        if args.eval_model == 'SimSiam':
            if args.dataset == 'CIFAR':
                model_file = 'SimSiam_CIFAR/SimSiam_CIFAR_final.pth'
                model = SimSiamVI.SimSiamVI(feature_dim=args.feature_dim, latent_dim=args.latent_dim, backbone=args.backbone)
            elif args.dataset == 'CIFAR100':
                model_file = 'SimSiam_CIFAR100/SimSiam_CIFAR100_final.pth'
                model = SimSiam.SimSiam(feature_dim=args.feature_dim, latent_dim=args.latent_dim, backbone=args.backbone)
            else:  # args.dataset == 'ImageNet':
                model_file = 'SimSiam_ImageNet/SimSiam_ImageNet_final.pth'
                model = SimSiam.SimSiam(feature_dim=args.feature_dim, latent_dim=args.latent_dim, backbone=args.backbone)

        elif args.eval_model == 'SimSiamVI':
            if args.dataset == 'CIFAR':
                model_file = 'SimSiamVI_CIFAR/SimSiamVI_CIFAR_final.pth'
                model = SimSiamVI.SimSiamVI(feature_dim=args.feature_dim, latent_dim=args.latent_dim, backbone=args.backbone)
            elif args.dataset == 'CIFAR100':
                model_file = 'SimSiamVI_CIFAR100/SimSiamVI_CIFAR100_final.pth'
                model = SimSiamVI.SimSiamVI(feature_dim=args.feature_dim, latent_dim=args.latent_dim, backbone=args.backbone)
            else:  # args.dataset == 'ImageNet':
                model_file = 'SimSiamVI_ImageNet/SimSiamVI_ImageNet_final.pth'
                model = SimSiamVI.SimSiamVI(feature_dim=args.feature_dim, latent_dim=args.latent_dim, backbone=args.backbone)

        else:  # args.eval_model == 'SimSiamVAE'
            if args.dataset == 'CIFAR':
                model_file = 'SimSiamVAE_CIFAR/SimSiamVAE_CIFAR_final.pth'
                model = SimSiamVAE.SimSiamVAE(feature_dim=args.feature_dim, latent_dim=args.latent_dim, variable_dim=args.variable_dim, backbone=args.backbone)
            elif args.dataset == 'CIFAR100':
                model_file = 'SimSiamVAE_CIFAR100/SimSiamVAE_CIFAR100_final.pth'
                model = SimSiamVAE.SimSiamVAE(feature_dim=args.feature_dim, latent_dim=args.latent_dim, variable_dim=args.variable_dim, backbone=args.backbone)
            else:  # args.dataset == 'ImageNet':
                model_file = 'SimSiamVAE_ImageNet2/SimSiamVAE_ImageNet_final.pth'
                model = SimSiamVAE.SimSiamVAE(feature_dim=args.feature_dim, latent_dim=args.latent_dim, variable_dim=args.variable_dim, backbone=args.backbone)

        model.load_state_dict(torch.load(model_file, map_location=device))
        print(model_file)
        print(model)
        model.to(device)

        train_features, train_labels = extract_features(model, train_dataloader, device)
        print('Finish extract train features')
        test_features, test_labels = extract_features(model, test_dataloader, device)
        print('Finish extract test features')
        classifier = train_classifier(classifier=args.classifier, features=train_features, labels=train_labels, num_classes=args.num_classes, device=device, epochs=args.epochs)
        evaluate_classifier(classifier=classifier, features=test_features, labels=test_labels, device=device)


if __name__ == "__main__":
    main()
