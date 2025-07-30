import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import numpy as np
import os
import torch.nn.functional as F
import matplotlib.pyplot as plt



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epochs = 50
batch_size = 32
learning_rate = 0.0005

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

class ImprovedResNetWithTSDF(nn.Module):
    def __init__(self):
        super().__init__()
        # LiDAR processing
        self.lidar_layer1 = ResidualBlock(1, 32)
        self.lidar_layer2 = ResidualBlock(32, 64)
        self.lidar_layer3 = ResidualBlock(64, 128)
        
        # TSDF processing
        self.tsdf_conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.tsdf_conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.tsdf_conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.tsdf_pool = nn.AdaptiveAvgPool2d((8, 8))
        self.tsdf_proj = nn.Linear(128 * 64, 360)
        
        # Fusion and output
        self.fusion_conv1 = nn.Conv1d(256, 128, 1)
        self.fusion_conv2 = nn.Conv1d(128, 64, 1)
        
        # Separate heads for min and max
        self.min_head = nn.Sequential(
            nn.Conv1d(64, 32, 1),
            nn.ReLU(),
            nn.Conv1d(32, 1, 1)
        )
        
        self.max_head = nn.Sequential(
            nn.Conv1d(64, 32, 1),
            nn.ReLU(),
            nn.Conv1d(32, 1, 1)
        )
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, lidar_data, tsdf_data):
        # Process LiDAR
        x_lidar = self.lidar_layer1(lidar_data)
        x_lidar = self.lidar_layer2(x_lidar)
        x_lidar = self.lidar_layer3(x_lidar)  # (batch, 128, 360)
        
        # Process TSDF
        x_tsdf = torch.relu(self.tsdf_conv1(tsdf_data))
        x_tsdf = torch.relu(self.tsdf_conv2(x_tsdf))
        x_tsdf = torch.relu(self.tsdf_conv3(x_tsdf))  # (batch, 128, 128, 128)
        
        # Pool and project TSDF
        x_tsdf = self.tsdf_pool(x_tsdf)  # (batch, 128, 8, 8)
        b, c, h, w = x_tsdf.shape
        x_tsdf_flat = x_tsdf.view(b, c * h * w)  # (batch, 128*64)
        x_tsdf_proj = self.tsdf_proj(x_tsdf_flat)  # (batch, 360)
        x_tsdf_proj = x_tsdf_proj.unsqueeze(1).expand(-1, 128, -1)  # (batch, 128, 360)
        
        # Fuse features
        x_fused = torch.cat([x_lidar, x_tsdf_proj], dim=1)  # (batch, 256, 360)
        x_fused = torch.relu(self.fusion_conv1(x_fused))
        x_fused = torch.relu(self.fusion_conv2(x_fused))
        x_fused = self.dropout(x_fused)
        
        # Generate predictions
        min_logits = self.min_head(x_fused).squeeze(1)  # (batch, 360)
        max_logits = self.max_head(x_fused).squeeze(1)  # (batch, 360)
        
        return min_logits, max_logits

def circular_distance(a, b, max_val=360):
    """Calculate circular distance between two angles"""
    diff = torch.abs(a - b)
    return torch.minimum(diff, max_val - diff)

def calculate_span(min_idx, max_idx):
    """Calculate proper span handling wraparound"""
    # If min <= max, it's a simple span
    # If min > max, it wraps around
    span = torch.where(min_idx <= max_idx, 
                      max_idx - min_idx + 1,
                      360 - min_idx + max_idx + 1)
    return span

def improved_min_max_loss(min_pred_logits, max_pred_logits, min_target, max_target, alpha=0.5):
    """
    Improved loss function with proper circular handling
    """
    # Convert to probabilities
    min_probs = F.softmax(min_pred_logits, dim=1)
    max_probs = F.softmax(max_pred_logits, dim=1)
    
    # Calculate expected indices
    angles = torch.arange(360, device=min_pred_logits.device, dtype=torch.float32)
    min_expected = torch.sum(min_probs * angles, dim=1)
    max_expected = torch.sum(max_probs * angles, dim=1)
    
    # Convert targets to float
    min_target_float = min_target.float()
    max_target_float = max_target.float()
    
    # Calculate circular distances
    min_dist = circular_distance(min_expected, min_target_float)
    max_dist = circular_distance(max_expected, max_target_float)
    
    # Calculate span loss
    target_span = calculate_span(min_target, max_target)
    predicted_span = calculate_span(min_expected.round().long(), max_expected.round().long())
    span_loss = F.mse_loss(predicted_span.float(), target_span.float())
    
    # Combine losses
    position_loss = F.mse_loss(min_dist, torch.zeros_like(min_dist)) + \
                   F.mse_loss(max_dist, torch.zeros_like(max_dist))
    
    # Add cross-entropy loss for sharper predictions
    ce_loss = F.cross_entropy(min_pred_logits, min_target) + \
              F.cross_entropy(max_pred_logits, max_target)
    
    return alpha * position_loss + (1-alpha) * ce_loss + 0.1 * span_loss

# Load data
base_dir = os.path.dirname(os.path.abspath(__file__))
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

# Use improved model
resnet = ImprovedResNetWithTSDF().to(device)
optimizer = optim.Adam(resnet.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

def train_model():
    loss_history = []
    
    for epoch in range(num_epochs):
        resnet.train()
        epoch_loss = 0.0
        
        for batch_idx, (distances, labels, _, tsdf) in enumerate(train_loader):
            distances = distances.unsqueeze(1).to(device)
            min_target, max_target = labels
            min_target = min_target.to(device)
            max_target = max_target.to(device)
            tsdf = tsdf.unsqueeze(1).to(device)
            
            optimizer.zero_grad()
            min_pred, max_pred = resnet(distances, tsdf)
            
            # Use improved loss function
            loss = improved_min_max_loss(min_pred, max_pred, min_target, max_target)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(resnet.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
        
        scheduler.step()
        avg_epoch_loss = epoch_loss / len(train_loader)
        loss_history.append(avg_epoch_loss)
        print(f'Epoch [{epoch+1}/{num_epochs}] Average Loss: {avg_epoch_loss:.4f}')
    
    return loss_history

def debug_test_model(loss_history):
    resnet.eval()
def debug_test_model(_):
    resnet.eval()
    total_samples = 0
    total_absolute_error = 0.0
    debug_samples = []
    
    with torch.no_grad():
        for batch_idx, (distances, labels, _, tsdf) in enumerate(test_loader):
            min_target = min_target.to(device)
            max_target = max_target.to(device)
            tsdf = tsdf.unsqueeze(1).to(device)
            
            min_pred, max_pred = resnet(distances, tsdf)
            min_predicted = torch.argmax(min_pred, dim=1)
            max_predicted = torch.argmax(max_pred, dim=1)
            
            # Calculate spans properly
            target_span = calculate_span(min_target, max_target)
            predicted_span = calculate_span(min_predicted, max_predicted)
            
            # Calculate absolute error
            absolute_error = torch.abs(predicted_span - target_span)
            
            # Debug: Print first few samples
            if batch_idx == 0:
                print("=== DEBUG: First 5 samples ===")
                for i in range(min(5, len(min_target))):
                    print(f"Sample {i}:")
                    print(f"  Target: min={min_target[i].item()}, max={max_target[i].item()}, span={target_span[i].item()}")
                    print(f"  Predicted: min={min_predicted[i].item()}, max={max_predicted[i].item()}, span={predicted_span[i].item()}")
                    print(f"  Error: {absolute_error[i].item():.2f} degrees")
                    print()
            
            # Store debug info for first batch
            if batch_idx == 0:
                for i in range(len(min_target)):
                    debug_samples.append({
                        'min_target': min_target[i].item(),
                        'max_target': max_target[i].item(),
                        'target_span': target_span[i].item(),
                        'min_pred': min_predicted[i].item(),
                        'max_pred': max_predicted[i].item(),
                        'pred_span': predicted_span[i].item(),
                        'error': absolute_error[i].item()
                    })
            
            total_absolute_error += absolute_error.sum().item()
            total_samples += min_target.size(0)

    # Calculate metrics
    avg_absolute_error = total_absolute_error / total_samples
    
    print(f'Average Absolute Error: {avg_absolute_error:.2f} degrees')
    print(f'Total samples: {total_samples}')
    
    # Analyze debug samples
    print("\n=== ERROR ANALYSIS ===")
    errors = [s['error'] for s in debug_samples]
    print(f"Min error: {min(errors):.2f}")
    print(f"Max error: {max(errors):.2f}")
    print(f"Median error: {sorted(errors)[len(errors)//2]:.2f}")
    
    # Check for specific issues
    large_errors = [s for s in debug_samples if s['error'] > 90]
    print(f"\nSamples with error > 90 degrees: {len(large_errors)}")
    for s in large_errors[:3]:
        print(f"  Target span: {s['target_span']:.1f}, Predicted span: {s['pred_span']:.1f}, Error: {s['error']:.1f}")
    
    return avg_absolute_error

def test_model(loss_history):
    resnet.eval()
    total_samples = 0
    total_absolute_error = 0.0
    total_relative_error = 0.0
    valid_relative_samples = 0
    
    with torch.no_grad():
        for distances, labels, _, tsdf in test_loader:
            distances = distances.unsqueeze(1).to(device)
            min_target, max_target = labels
            min_target = min_target.to(device)
            max_target = max_target.to(device)
            tsdf = tsdf.unsqueeze(1).to(device)
            
            min_pred, max_pred = resnet(distances, tsdf)
            min_predicted = torch.argmax(min_pred, dim=1)
            max_predicted = torch.argmax(max_pred, dim=1)
            
            # Calculate target distance (ground truth)
            distance_target = (max_target - min_target + 360) % 360
            # Calculate predicted distance
            distance_predicted = (max_predicted - min_predicted + 360) % 360
            
            # Calculate absolute error (in degrees)
            absolute_error = torch.abs(distance_predicted - distance_target)
            total_absolute_error += absolute_error.sum().item()
            
            # Calculate relative error only for non-zero targets
            non_zero_mask = distance_target > 0
            if non_zero_mask.any():
                relative_error = absolute_error[non_zero_mask] / distance_target[non_zero_mask]
                # Cap relative error at 2.0 to prevent extreme outliers
                relative_error = torch.clamp(relative_error, 0, 2.0)
                total_relative_error += relative_error.sum().item()
                valid_relative_samples += non_zero_mask.sum().item()
            
            total_samples += min_target.size(0)

    # Calculate metrics
    avg_absolute_error = total_absolute_error / total_samples
    
    if valid_relative_samples > 0:
        avg_relative_error = total_relative_error / valid_relative_samples
        accuracy = (1 - avg_relative_error) * 100.0
    else:
        accuracy = 0.0
        avg_relative_error = 0.0
    
    print(f'Test Accuracy: {accuracy:.2f}%')
    print(f'Average Absolute Error: {avg_absolute_error:.2f} degrees')
    print(f'Average Relative Error: {avg_relative_error:.4f}')
    print(f'Valid samples for relative error: {valid_relative_samples}/{total_samples}')
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(num_epochs), loss_history, label='Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.show()
    
    return accuracy, avg_relative_error

# Alternative: Use a more robust accuracy metric
def test_model_robust(loss_history):
    resnet.eval()
    total_samples = 0
    correct_predictions = 0
    total_absolute_error = 0.0
    
    with torch.no_grad():
        for distances, labels, _, tsdf in test_loader:
            distances = distances.unsqueeze(1).to(device)
            min_target, max_target = labels
            min_target = min_target.to(device)
            max_target = max_target.to(device)
            tsdf = tsdf.unsqueeze(1).to(device)
            
            min_pred, max_pred = resnet(distances, tsdf)
            min_predicted = torch.argmax(min_pred, dim=1)
            max_predicted = torch.argmax(max_pred, dim=1)
            
            # Calculate target distance (ground truth)
            distance_target = (max_target - min_target + 360) % 360
            # Calculate predicted distance
            distance_predicted = (max_predicted - min_predicted + 360) % 360
            
            # Calculate absolute error
            absolute_error = torch.abs(distance_predicted - distance_target)
            total_absolute_error += absolute_error.sum().item()
            
            # Count "correct" predictions (within tolerance)
            tolerance = 10  # degrees
            correct_predictions += (absolute_error <= tolerance).sum().item()
            
            total_samples += min_target.size(0)

    # Calculate metrics
    avg_absolute_error = total_absolute_error / total_samples
    tolerance_accuracy = (correct_predictions / total_samples) * 100.0
    
    print(f'Tolerance Accuracy (±{tolerance}°): {tolerance_accuracy:.2f}%')
    print(f'Average Absolute Error: {avg_absolute_error:.2f} degrees')
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(num_epochs), loss_history, label='Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.show()
    
    return tolerance_accuracy, avg_absolute_error

if __name__ == "__main__":
    loss_history = train_model()
    tolerance_accuracy, avg_absolute_error = test_model_robust(loss_history)
    model_path = os.path.join(data_dir, "improved_resnet_lidar_model_tsdf_min_max.pth")
    torch.save(resnet.state_dict(), model_path)
    print(f'Model saved to {model_path}')