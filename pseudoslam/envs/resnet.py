import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset    
import torch.optim as optim
import numpy as np
import os
import matplotlib.pyplot as plt


"""This script implements a ResNet for classifying LIDAR data as obstacles or non-obstacles using PyTorch."""

# Configure device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Basic parameters
num_epochs = 100
batch_size = 32
learning_rate = 0.0005

# Creating custom dataset class for LIDAR data (also used for label datasets)
class LidarDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        distances = torch.tensor(self.data[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return distances, label
    
# Load the datasets
# Set path to be data directory
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, "data")
# Load training and testing data (LIDAR and labels)
training_lidar = np.load(os.path.join(data_dir, "training_lidar.npy")) 
training_labels = np.load(os.path.join(data_dir, "training_labels.npy")) 
testing_lidar = np.load(os.path.join(data_dir, "testing_lidar.npy")) 
labels_test = np.load(os.path.join(data_dir, "testing_labels.npy"))  

# Create datasets and dataloaders
train_dataset = LidarDataset(training_lidar, training_labels)
test_dataset = LidarDataset(testing_lidar, labels_test)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)

        # If the input and output channels are different, we need a projection
        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = ResidualBlock(1, 16)
        self.layer2 = ResidualBlock(16, 32)
        self.layer3 = ResidualBlock(32, 64)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)  # Global average pooling
        self.fc = nn.Linear(64, 1)  # Final fully connected layer for

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.global_avg_pool(x)  # Shape: (batch_size, 64, 1)
        return self.fc(x.squeeze(-1))  # Shape: (batch_size, 1)
    


resnet = ResNet().to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(resnet.parameters(), lr=learning_rate, weight_decay=1e-4)

n_total_steps = len(train_loader)
loss_history = []
for epoch in range(num_epochs):
    resnet.train()
    epoch_loss = 0.0
    for i, (distances, labels) in enumerate(train_loader):
        distances = distances.unsqueeze(1).to(device)  # Shape: (batch, 1, length)
        labels = labels.to(device)
        # Reduce labels to shape (batch, 1)
        if labels.dim() == 2 and labels.shape[1] == distances.shape[-1]:
            labels = labels[:, 0]  # Take the first value if label is repeated per distance
        labels = labels.float().unsqueeze(1)  # Shape: (batch, 1)
        logits = resnet(distances)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')
    loss_history.append(epoch_loss / n_total_steps)

print('Finished training')

n_correct = 0

with torch.no_grad():
    total = 0
    for distances, labels in test_loader:
        distances = distances.unsqueeze(1).to(device)  # Shape: (batch, 1, length)
        # Reduce labels to shape (batch, 1)
        labels = labels.to(device)
        if labels.dim() == 2 and labels.shape[1] == distances.shape[-1]:
            labels = labels[:, 0]
        labels = labels.float().unsqueeze(1)  # Shape: (batch, 1)
        outputs = resnet(distances)
        predicted = (torch.sigmoid(outputs) > 0.5).float()
        total += labels.size(0)
        n_correct += (predicted == labels).sum().item()

    accuracy = 100 * n_correct / total
    print(f'Accuracy of the network: {accuracy:.4f} %')
    plt.plot(range(num_epochs), loss_history, label='Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.show()

        



