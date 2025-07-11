import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset    
import torch.optim as optim
import numpy as np
import os
from torchvision.ops import sigmoid_focal_loss
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
    def __init__(self, data, labels, mask=None):
        self.data = data
        self.labels = labels
        self.mask = mask if mask is not None else np.ones_like(labels)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        distances = torch.tensor(self.data[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        mask = torch.tensor(self.mask[idx], dtype=torch.float32)
        return distances, label, mask
    
# Load the datasets
# Set path to be data directory
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, "data")
# Load training and testing data (LIDAR and labels)
training_lidar = np.load(os.path.join(data_dir, "training_lidar.npy")) 
training_labels = np.load(os.path.join(data_dir, "training_labels.npy")) 
testing_lidar = np.load(os.path.join(data_dir, "testing_lidar.npy")) 
labels_test = np.load(os.path.join(data_dir, "testing_labels.npy"))  

# Masking 
training_mask = np.random.binomial(1, 0.9, training_labels.shape).astype(np.float32)
testing_mask = np.ones_like(labels_test, dtype=np.float32)  # all valid in testing

# Create datasets and dataloaders
train_dataset = LidarDataset(training_lidar, training_labels, training_mask)
test_dataset = LidarDataset(testing_lidar, labels_test, testing_mask)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(out_channels)

        # If the input and output channels are different, we need a projection
        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out += residual
        out = self.relu2(out)
        return out

class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = ResidualBlock(1, 16)
        self.layer2 = ResidualBlock(16, 32)
        self.layer3 = ResidualBlock(32, 64)
        # Remove global pooling!
        self.classifier = nn.Conv1d(64, 1, kernel_size=1)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.classifier(x)  # Shape: (batch_size, 1, 360)
        return x.squeeze(1)     # Shape: (batch_size, 360)


resnet = ResNet().to(device)

def criterion(logits, targets):
    # logits: (batch, 360), targets: (batch, 360)
    return sigmoid_focal_loss(logits, targets, reduction="mean")

optimizer = optim.Adam(resnet.parameters(), lr=learning_rate, weight_decay=1e-4)

n_total_steps = len(train_loader)
loss_history = []
for epoch in range(num_epochs):
    resnet.train()
    epoch_loss = 0.0
    for i, (distances, labels, mask) in enumerate(train_loader):
        distances = distances.unsqueeze(1).to(device)  # (B, 1, 360)
        labels = labels.to(device)                     # (B, 360)
        mask = mask.to(device)                         # (B, 360)

        logits = resnet(distances)                     # (B, 360)
        raw_loss = sigmoid_focal_loss(logits, labels, reduction="none")  # (B, 360)

        # Apply mask to ignore some loss entries
        masked_loss = raw_loss * mask
        loss = masked_loss.sum() / (mask.sum() + 1e-8)  # normalize only over valid parts

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
    for distances, labels, mask in test_loader:
        distances = distances.unsqueeze(1).to(device)
        labels = labels.to(device).float()
        mask = mask.to(device)

        outputs = resnet(distances)
        predicted = (torch.sigmoid(outputs) > 0.5).float()

        total += mask.sum().item()
        n_correct += ((predicted == labels) * mask).sum().item()

    accuracy = 100 * n_correct / (total + 1e-8)
    print(f'Accuracy of the network (masked): {accuracy:.4f} %')
    plt.plot(range(num_epochs), loss_history, label='Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.show()

# Save the trained model
model_path = os.path.join(data_dir, "resnet_lidar_model.pth")
torch.save(resnet.state_dict(), model_path)



