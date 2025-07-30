import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset    
import torch.optim as optim
import numpy as np


"""This script implements a 1D CNN for classifying LIDAR data as obstacles or non-obstacles using PyTorch."""

# configure device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# parameters
num_epochs = 10
batch_size = 60
learning_rate = 0.001

# data transformation from array to tensor

class LidarDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        distances = self.data[idx]
        label = self.labels[idx]
        return distances, label
    
class TestDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        distances = self.data[idx]
        label = self.labels[idx]
        return distances, label
    
# Load the dataset
distances_train = np.array(
    [14, 15, 15, 15, 14, 14, 15, 15, 16, 16, 17, 18, 18, 19,
    68, 68, 67, 67, 66, 66, 66, 65, 65, 65, 65, 64, 64, 64,
    64, 64, 64, 64, 63, 63, 63, 63, 63, 63, 63, 63, 64, 64,
    64, 64, 64, 64, 64, 65, 65, 65, 65, 66, 66, 66, 67, 67,
    68, 68, 69, 69, 70, 70, 71, 71, 72, 73, 74, 74, 75, 76,
    77, 78, 79, 80, 81, 82, 83, 85, 86, 87, 89, 90, 92, 94,
    96, 97, 99, 102, 104, 106, 109, 112, 115, 118, 121, 125, 128, 156,
    159, 158, 157, 155, 154, 153, 152, 151, 150, 149, 148, 147, 147, 146,
    145, 145, 144, 144, 143, 143, 143, 132, 132, 132, 132, 132, 131, 131,
    131, 131, 132, 132, 142, 142, 142, 143, 143, 143, 127, 116, 107, 99,
    92, 87, 82, 77, 73, 70, 66, 63, 61, 58, 56, 54, 52, 50,
    49, 47, 46, 45, 43, 42, 41, 40, 39, 38, 38, 37, 36, 35,
    35, 34, 33, 33, 32, 32, 31, 31, 29, 29, 30, 200, 200, 200,
    200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 199, 198,
    197, 196, 195, 194, 193, 192, 191, 190, 189, 189, 188, 188, 187, 187,
    186, 186, 176, 186, 176, 176, 176, 176, 176, 176, 176, 176, 177, 177,
    178, 178, 179, 179, 180, 181, 181, 172, 25, 24, 24, 24, 24, 24,
    24, 25, 25, 25, 25, 25, 26, 26, 26, 27, 27, 27, 27, 28,
    28, 28, 29, 29, 30, 30, 31, 31, 32, 32, 33, 33, 34, 35,
    35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 49, 50,
    52, 51, 51, 50, 50, 50, 49, 49, 49, 49, 48, 48, 48, 48,
    47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47,
    47, 47, 47, 47, 47, 47, 47, 47, 47, 48, 48, 48, 48, 48,
    49, 49, 49, 49, 50, 50, 50, 51, 51, 52, 52, 52, 53, 18,
    18, 15, 15, 15, 15, 15, 15, 15, 15, 14, 14, 14, 14, 14,
    15, 14, 14, 14, 14, 13, 13, 13, 14, 14]
, dtype=np.float32)

labels_train = np.array(
    [1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  
    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  
    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  
    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  
    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  
    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  
    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  
    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  
    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  
    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  
    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  
    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  
    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  
    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  
    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  
    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  
    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  
    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  
    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  
    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  
    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  
    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  
    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  
    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  
    1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  
    1,  1,  1,  1,  1,  1,  1,  1,  1,  1], dtype=np.float32)

distances_test = np.array([
    40, 40, 40, 40, 41, 41, 41, 41, 42, 42, 42, 42, 43, 43, 43, 44, 44, 45, 45, 46,
    46, 47, 47, 48, 48, 49, 50, 50, 51, 52, 53, 53, 54, 55, 56, 57, 57, 57, 56, 55,
    54, 54, 53, 52, 52, 51, 51, 50, 49, 49, 49, 48, 48, 47, 47, 47, 46, 46, 46, 45,
    45, 45, 45, 45, 44, 44, 44, 44, 44, 44, 43, 43, 43, 43, 43, 43, 43, 43, 43, 141,
    141, 141, 142, 142, 142, 142, 143, 143, 143, 144, 145, 145, 146, 146, 147, 148,
    149, 150, 151, 152, 153, 154, 155, 156, 158, 159, 160, 162, 51, 50, 51, 52, 52,
    53, 53, 54, 55, 56, 57, 57, 58, 59, 60, 61, 62, 64, 65, 66, 67, 69, 70, 72, 74,
    76, 78, 80, 82, 84, 87, 90, 93, 96, 99, 103, 107, 112, 117, 122, 128, 134, 142,
    145, 144, 144, 143, 142, 142, 141, 141, 141, 140, 140, 140, 139, 139, 139, 139,
    139, 139, 139, 139, 139, 139, 140, 140, 140, 141, 141, 141, 142, 142, 143, 144,
    144, 145, 146, 147, 147, 148, 149, 150, 152, 153, 154, 155, 157, 158, 160, 173,
    163, 165, 167, 169, 171, 173, 175, 190, 171, 168, 164, 161, 158, 155, 153, 150,
    148, 145, 143, 141, 139, 137, 135, 134, 132, 131, 129, 128, 126, 125, 124, 123,
    122, 121, 120, 119, 118, 117, 116, 116, 115, 114, 114, 113, 112, 112, 112, 111,
    111, 110, 110, 110, 110, 109, 67, 65, 64, 63, 62, 62, 61, 61, 61, 61, 62, 62,
    62, 64, 67, 121, 121, 111, 112, 112, 113, 114, 114, 115, 115, 111, 106, 101, 97,
    93, 90, 87, 84, 81, 79, 76, 74, 72, 70, 68, 67, 65, 64, 62, 61, 60, 59, 57, 56,
    55, 54, 54, 53, 52, 51, 50, 50, 49, 48, 48, 47, 47, 46, 46, 45, 45, 44, 44, 43,
    43, 43, 42, 42, 42, 42, 41, 41, 41, 41, 40, 40, 40, 40, 40, 40, 39, 39, 39, 39,
    39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 40, 40
], dtype=np.float32)

labels_test = np.array([
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
    1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0
], dtype=np.float32)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
])  

train_dataset = LidarDataset(distances_train, labels_train)

test_dataset = TestDataset(distances_test, labels_test)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

classes = ('0', '1')

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, stride=1, padding=2),  
            nn.BatchNorm1d(16),
            nn.ReLU(),

            nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Conv1d(64, 1, kernel_size = 1)
        )

    def forward(self, x):
        logits = self.net(x)
        return logits[:, 0,:] 
    


cnn = CNN().to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(cnn.parameters(), lr=learning_rate, weight_decay=1e-4)

n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    cnn.train()
    for i, (distances, labels) in enumerate(train_loader):
        distances = distances.unsqueeze(1).unsqueeze(2).to(device)  # Shape: (batch, 1, length)
        labels = labels.to(device).unsqueeze(1)  # Shape: (batch, 1)
        logits = cnn(distances)
        loss = criterion(logits, labels.float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')


print('Finished training')

with torch.no_grad():
    total = 0
    n_correct = 0
    for distances, labels in test_loader:
        distances = distances.unsqueeze(1).unsqueeze(2).to(device)  # Shape: (batch, 1, length)
        labels = labels.to(device).unsqueeze(1)  # Shape: (batch, 1)
        outputs = cnn(distances)
        predicted = (torch.sigmoid(outputs) > 0.5).float()
        total += labels.size(0)
        n_correct += (predicted == labels).sum().item()

    accuracy = 100 * n_correct / total
    print(f'Accuracy of the network: {accuracy} %')
        



