import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


def extract_features(model, dataloader, device):
    model.eval()
    features = []
    labels = []

    encoder = (
            getattr(model, "encoder_online", None) or
            getattr(model, "encoder", None)
    )

    with torch.no_grad():
        for imgs, targets in tqdm(dataloader, desc="Extracting"):
            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with torch.amp.autocast(device_type='cuda'):
                y = encoder(imgs)

            features.append(y.contiguous().cpu())
            labels.append(targets.cpu())

    features = torch.cat(features).numpy()
    labels = torch.cat(labels).numpy()
    return features, labels


class Classifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        x = self.fc(x)
        return x


def train_classifier(features, labels, num_classes, device, epochs=100):
    classifier = Classifier(features.shape[1], num_classes).to(device)

    optimizer = optim.Adam(classifier.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    dataset = TensorDataset(torch.tensor(features).float(), torch.tensor(labels).long())
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

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
