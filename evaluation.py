import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


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
