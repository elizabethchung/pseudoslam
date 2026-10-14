import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import numpy as np
import os
from torchvision.ops import sigmoid_focal_loss
import torch.nn.functional as F
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epochs = 50
batch_size = 32
learning_rate = 0.001

torch.autograd.set_detect_anomaly(True)
torch.cuda.empty_cache()

def lidar_to_tsdf_obstacles(lidar_data, obstacle_predictions, img_dim=128, max_range=10.0):
    """ This function converts LIDAR data and obstacle predictions into a TSDF map by projecting the data into a 2D grid."""
    batch_size = lidar_data.shape[0]
    tsdf_maps = np.zeros((batch_size, img_dim, img_dim))
    center = img_dim // 2
    y_coords, x_coords = np.meshgrid(np.arange(img_dim), np.arange(img_dim))
    angles = np.linspace(0, 2*np.pi, 360, endpoint=False)
    for batch_idx in range(batch_size):
        distances = lidar_data[batch_idx]
        obstacles = obstacle_predictions[batch_idx]
        for _, (angle, distance, is_obstacle) in enumerate(zip(angles, distances, obstacles)):
            if 0 < distance < max_range:
                x_world = distance * np.cos(angle)
                y_world = distance * np.sin(angle)
                x_grid = int(center + x_world * (img_dim // 2) / max_range)
                y_grid = int(center + y_world * (img_dim // 2) / max_range)
                if 0 <= x_grid < img_dim and 0 <= y_grid < img_dim:
                    dist_to_point = np.sqrt((x_coords - x_grid)**2 + (y_coords - y_grid)**2)
                    if is_obstacle:
                        obstacle_sdf = -np.maximum(0, 3 - dist_to_point)
                        tsdf_maps[batch_idx] = np.minimum(tsdf_maps[batch_idx], obstacle_sdf)
                    else:
                        free_sdf = np.maximum(0, dist_to_point - 1)
                        tsdf_maps[batch_idx] = np.maximum(tsdf_maps[batch_idx], free_sdf)
    return tsdf_maps

def labels_to_min_max(labels):
    labels = np.atleast_2d(labels)
    min_indices, max_indices = [], []
    for row in labels:
        ones_indices = np.where(row == 1)[0]
        if len(ones_indices) == 0:
            min_indices.append(0)
            max_indices.append(row.shape[0] - 1)
        elif len(ones_indices) == 1:
            min_indices.append(ones_indices[0])
            max_indices.append(ones_indices[0])
        else:
            sorted_indices = np.sort(ones_indices)
            gaps = np.diff(sorted_indices)
            wrap_gap = (row.shape[0] - sorted_indices[-1]) + sorted_indices[0]
            max_gap_idx = np.argmax(gaps)
            max_gap = gaps[max_gap_idx] if gaps.size > 0 else -1
            if wrap_gap > max_gap:
                min_indices.append(sorted_indices[0])
                max_indices.append(sorted_indices[-1])
            else:
                min_indices.append(sorted_indices[max_gap_idx + 1])
                max_indices.append(sorted_indices[max_gap_idx])
    return np.array(min_indices), np.array(max_indices)

class LidarTSDFDataset(Dataset):
    def __init__(self, data, labels, mask=None):
        self.data = data
        self.mask = mask if mask is not None else np.ones((len(data), 360))
        if isinstance(labels, tuple):
            self.min_indices, self.max_indices = labels
            binary_labels = np.zeros((len(self.min_indices), 360))
            for i, (min_idx, max_idx) in enumerate(zip(self.min_indices, self.max_indices)):
                if min_idx <= max_idx:
                    binary_labels[i, min_idx:max_idx+1] = 1
                else:
                    binary_labels[i, min_idx:] = 1
                    binary_labels[i, :max_idx+1] = 1
            self.labels = binary_labels
        else:
            self.labels = labels
            self.min_indices, self.max_indices = labels_to_min_max(labels)
        self.tsdf_maps = lidar_to_tsdf_obstacles(self.data, (self.labels > 0.5).astype(np.float32))
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        distances = torch.tensor(self.data[idx], dtype=torch.float32)
        min_idx = torch.tensor(self.min_indices[idx], dtype=torch.long)
        max_idx = torch.tensor(self.max_indices[idx], dtype=torch.long)
        mask = torch.tensor(self.mask[idx], dtype=torch.float32)
        tsdf = torch.tensor(self.tsdf_maps[idx], dtype=torch.float32)
        return distances, (min_idx, max_idx), mask, tsdf

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.conv3 = nn.Conv1d(out_channels, out_channels, 3, padding=1)
        self.bn3 = nn.BatchNorm1d(out_channels)
        self.shortcut = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.relu = nn.ReLU()
    def forward(self, x):
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.relu(self.bn3(self.conv3(out)))
        out = out + residual
        return self.relu(out)

class ResNetWithTSDF(nn.Module):
    def __init__(self):
        super().__init__()
        self.lidar_layer1 = ResidualBlock(1, 16)
        self.lidar_layer2 = ResidualBlock(16, 32)
        self.lidar_layer3 = ResidualBlock(32, 64)
        self.tsdf_conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.tsdf_conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.tsdf_conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.tsdf_proj = nn.Linear(128*128, 360)
        self.fusion_conv = nn.Conv1d(128, 64, 1)
        self.min_classifier = nn.Conv1d(64, 360, 1)
        self.max_classifier = nn.Conv1d(64, 360, 1)
    def forward(self, lidar_data, tsdf_data):
        x_lidar = self.lidar_layer1(lidar_data)
        x_lidar = self.lidar_layer2(x_lidar)
        x_lidar = self.lidar_layer3(x_lidar)
        x_tsdf = torch.relu(self.tsdf_conv1(tsdf_data))
        x_tsdf = torch.relu(self.tsdf_conv2(x_tsdf))
        x_tsdf = torch.relu(self.tsdf_conv3(x_tsdf))
        b, c, _, _ = x_tsdf.shape
        x_tsdf_flat = x_tsdf.view(b, c, -1)
        x_tsdf_proj = self.tsdf_proj(x_tsdf_flat)
        # x_lidar: (batch, 64, 360), x_tsdf_proj: (batch, 64, 360)
        x_fused = torch.cat([x_lidar, x_tsdf_proj], dim=1)  # (batch, 128, 360)
        x_fused = self.fusion_conv(x_fused)  # (batch, 64, 360)
        min_logits = self.min_classifier(x_fused)  # (batch, 360, 360)
        max_logits = self.max_classifier(x_fused)  # (batch, 360, 360)
        # Take diagonal logits as the prediction for each angle
        min_pred = torch.diagonal(min_logits, dim1=1, dim2=2)  # (batch, 360)
        max_pred = torch.diagonal(max_logits, dim1=1, dim2=2)  # (batch, 360)
        return min_pred, max_pred

def min_max_loss(min_pred, max_pred, min_target, max_target):
    # min_pred, max_pred: (batch, 360), min_target, max_target: (batch,)
    min_loss = sigmoid_focal_loss(min_pred, F.one_hot(min_target, num_classes=360).float(), reduction="mean")
    max_loss = sigmoid_focal_loss(max_pred, F.one_hot(max_target, num_classes=360).float(), reduction="mean")
    return min_loss + max_loss

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # pseudoslam/envs (data/ lives there)
data_dir = os.path.join(base_dir, "data")
training_lidar = np.load(os.path.join(data_dir, "training_lidar.npy"))
training_labels = np.load(os.path.join(data_dir, "training_labels.npy"))
testing_lidar = np.load(os.path.join(data_dir, "testing_lidar.npy"))
labels_test = np.load(os.path.join(data_dir, "testing_labels.npy"))

training_min_indices, training_max_indices = labels_to_min_max(training_labels)
testing_min_indices, testing_max_indices = labels_to_min_max(labels_test)
training_mask = np.random.binomial(1, 0.9, training_labels.shape).astype(np.float32)
testing_mask = np.ones_like(labels_test, dtype=np.float32)

train_dataset = LidarTSDFDataset(training_lidar, (training_min_indices, training_max_indices), training_mask)
test_dataset = LidarTSDFDataset(testing_lidar, (testing_min_indices, testing_max_indices), testing_mask)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

resnet = ResNetWithTSDF().to(device)
optimizer = optim.Adam(resnet.parameters(), lr=learning_rate, weight_decay=1e-4)

def train_model():
    loss_history = []
    step = 0
    for epoch in range(num_epochs):
        resnet.train()
        for batch_idx, (distances, labels, _, tsdf) in enumerate(train_loader):
            distances = distances.unsqueeze(1).to(device)
            min_target, max_target = labels
            min_target = min_target.to(device)
            max_target = max_target.to(device)
            tsdf = tsdf.unsqueeze(1).to(device)
            min_pred, max_pred = resnet(distances, tsdf)
            loss = min_max_loss(min_pred, max_pred, min_target, max_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss = 0.0
            epoch_loss = epoch_loss + loss.item()
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
            step += 1
        loss_history.append(epoch_loss / len(train_loader))
    return loss_history

def test_model(loss_history):
    resnet.eval()
    n_correct_min = 0
    n_correct_max = 0
    total = 0
    for distances, labels, _, tsdf in test_loader:
        distances = distances.unsqueeze(1).to(device)
        min_target, max_target = labels
        min_target = min_target.to(device)
        max_target = max_target.to(device)
        tsdf = tsdf.unsqueeze(1).to(device)
        min_pred, max_pred = resnet(distances, tsdf)
        min_predicted = torch.argmax(min_pred, dim=1)
        max_predicted = torch.argmax(max_pred, dim=1)
        batch_size = min_target.size(0)
        total += batch_size
        
    min_accuracy = 100 * n_correct_min / total
    max_accuracy = 100 * n_correct_max / total
    print(f'Min Index Accuracy: {min_accuracy:.2f} %')
    print(f'Max Index Accuracy: {max_accuracy:.2f} %')
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(num_epochs), loss_history, label='Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.bar(['Min Index', 'Max Index'], [min_accuracy, max_accuracy])
    plt.ylabel('Accuracy (%)')
    plt.title('Test Accuracy')
    plt.tight_layout()
    plt.show()
    return min_accuracy, max_accuracy

if __name__ == "__main__":
    loss_history = train_model()
    min_accuracy, max_accuracy = test_model(loss_history)
    model_path = os.path.join(data_dir, "resnet_lidar_model_tsdf_min_max.pth")
    torch.save(resnet.state_dict(), model_path)
    print(f'Model saved to {model_path}')
