import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset    
import torch.optim as optim
import numpy as np
import os
from torchvision.ops import sigmoid_focal_loss
import matplotlib.pyplot as plt
import torch.nn.functional as F


"""Added TSDF to my ReNet"""

# Configure device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Basic parameters
num_epochs = 50
batch_size = 32
learning_rate = 0.0005

# xDim function computes the signed distance function (SDF) for the x dimension
# SDF is a representation of the distance to the nearest surface, where negative values indicate inside an object and positive values indicate outside.
def xDim(coors, bias_set, trunc_thresh):
    relu_op = lambda x: np.maximum(x, 0)
    x = coors[:, :, 0]
    a = bias_set[:, 0][..., None]
    b = bias_set[:, 1][..., None]
    sdf = np.minimum(
        relu_op(x - a + trunc_thresh) - relu_op(x - a - trunc_thresh) - trunc_thresh,
        -relu_op(x - b + trunc_thresh) + relu_op(x - b - trunc_thresh) + trunc_thresh,
    )
    return sdf

# yDim function computes the signed distance function (SDF) for the y dimension
def yDim(coors, bias_set, trunc_thresh):
    relu_op = lambda x: np.maximum(x, 0)
    x = coors[:, :, 1]
    a = bias_set[:, 2][..., None]
    b = bias_set[:, 3][..., None]
    sdf = np.minimum(
        relu_op(x - a + trunc_thresh) - relu_op(x - a - trunc_thresh) - trunc_thresh,
        -relu_op(x - b + trunc_thresh) + relu_op(x - b - trunc_thresh) + trunc_thresh,
    )
    return sdf

# Function to convert LIDAR data and obstacle predictions to TSDF maps
def lidar_to_tsdf_obstacles(lidar_data, obstacle_predictions, img_dim=128, max_range=10.0):
    """
    Convert LIDAR data and obstacle predictions to TSDF focused on obstacles.

    """
    batch_size = lidar_data.shape[0]
    tsdf_maps = np.zeros((batch_size, img_dim, img_dim))
    
    # Create coordinate grid
    center = img_dim // 2
    y_coords, x_coords = np.meshgrid(np.arange(img_dim), np.arange(img_dim))
    
    for batch_idx in range(batch_size):
        distances = lidar_data[batch_idx]
        obstacles = obstacle_predictions[batch_idx]
        
        # Convert LIDAR to Cartesian coordinates
        angles = np.linspace(0, 2*np.pi, 360, endpoint=False)
        
        for i, (angle, distance, is_obstacle) in enumerate(zip(angles, distances, obstacles)):
            if distance > 0 and distance < max_range:
                # Convert to grid coordinates
                x_world = distance * np.cos(angle)
                y_world = distance * np.sin(angle)
                
                # Scale to grid coordinates
                x_grid = int(center + x_world * (img_dim // 2) / max_range)
                y_grid = int(center + y_world * (img_dim // 2) / max_range)
                
                if 0 <= x_grid < img_dim and 0 <= y_grid < img_dim:
                    # Calculate distance to this point for all grid cells
                    dist_to_point = np.sqrt((x_coords - x_grid)**2 + (y_coords - y_grid)**2)
                    
                    if is_obstacle:
                        # For obstacles, create negative SDF (inside object)
                        obstacle_sdf = -np.maximum(0, 3 - dist_to_point)  # 3-pixel radius
                        tsdf_maps[batch_idx] = np.minimum(tsdf_maps[batch_idx], obstacle_sdf)
                    else:
                        # For free space, create positive SDF (outside object)
                        free_sdf = np.maximum(0, dist_to_point - 1)  # 1-pixel clearance
                        tsdf_maps[batch_idx] = np.maximum(tsdf_maps[batch_idx], free_sdf)
    
    return tsdf_maps

def extract_obstacle_features_from_tsdf(tsdf_map, threshold=-0.5):
    batch_size, img_dim, _ = tsdf_map.shape
    obstacle_mask = tsdf_map < threshold
    
    features = []
    for batch_idx in range(batch_size):
        mask = obstacle_mask[batch_idx]
        
        # Extract features from obstacles
        obstacle_count = np.sum(mask)
        obstacle_density = obstacle_count / (img_dim * img_dim)
        
        # Find obstacle regions
        if obstacle_count > 0:
            obstacle_coords = np.where(mask)
            centroid_x = np.mean(obstacle_coords[0])
            centroid_y = np.mean(obstacle_coords[1])
            
            # Calculate spread/concentration
            spread_x = np.std(obstacle_coords[0])
            spread_y = np.std(obstacle_coords[1])
        else:
            centroid_x = centroid_y = img_dim // 2
            spread_x = spread_y = 0
        
        batch_features = [
            obstacle_density,
            centroid_x / img_dim,  # Normalized
            centroid_y / img_dim,  # Normalized
            spread_x / img_dim,    # Normalized
            spread_y / img_dim,    # Normalized
        ]
        features.append(batch_features)
    
    return np.array(features)

def params_to_map_obstacle_focused(
    sdf_params,
    prob_params,
    coorVec,
    obstacle_mask=None,
    training_flag=True,
    gate_flag=True,
    batch_size=128,
    trunc_thresh=128,
    img_dim=128,
):
    coorVec_tmp = np.repeat(coorVec[np.newaxis, :, :], batch_size, axis=0)
    y_map = np.tile(np.array([-1]), (1, img_dim * img_dim))
    y_room_maps = np.zeros(
        (sdf_params.shape[0], img_dim * img_dim, sdf_params.shape[1] // 4)
    )
    
    for room in range(sdf_params.shape[1] // 4):
        bias_room = sdf_params[:, room * 4 + 0 : room * 4 + 4]
        sdfMap_room = np.minimum(
            xDim(coorVec_tmp, bias_room, trunc_thresh),
            yDim(coorVec_tmp, bias_room, trunc_thresh),
        )
        
        # Apply obstacle mask if provided
        if obstacle_mask is not None:
            obstacle_weight = obstacle_mask.reshape(batch_size, img_dim * img_dim)
            sdfMap_room = sdfMap_room * obstacle_weight
        
        if gate_flag:
            gate_room = prob_params[:, room, None]
            if not training_flag:
                gate_room = float(gate_room > 0.5)
            y_room_maps[:, :, room] = gate_room * (sdfMap_room + trunc_thresh)
            y_map = np.maximum(y_map, y_room_maps[:, :, room])
        else:
            y_room_maps[:, :, room] = sdfMap_room + trunc_thresh
            y_map = np.maximum(y_map, y_room_maps[:, :, room])

    y_map = y_map - trunc_thresh
    y_map = y_map.reshape(batch_size, img_dim, img_dim, 1)
    y_room_maps = y_room_maps - trunc_thresh
    y_room_maps = y_room_maps.reshape(
        batch_size, img_dim, img_dim, sdf_params.shape[1] // 4
    )
    
    return y_map, None, y_room_maps

# Enhanced Dataset class that includes TSDF processing
class LidarTSDFDataset(Dataset):
    def __init__(self, data, labels, mask=None, use_tsdf=True):
        self.data = data
        self.mask = mask if mask is not None else np.ones((len(data), 360))
        self.use_tsdf = use_tsdf
        
        # Handle different label formats
        if isinstance(labels, tuple):
            # New format: (min_indices, max_indices)
            self.min_indices, self.max_indices = labels
            self.labels = self._indices_to_binary_labels(self.min_indices, self.max_indices)
        else:
            # Old format: binary labels array
            self.labels = labels
            self.min_indices, self.max_indices = labels_to_min_max(labels)
        
        # Pre-compute TSDF maps for efficiency
        if use_tsdf:
            print("Pre-computing TSDF maps...")
            self.tsdf_maps = self._precompute_tsdf_maps()
        else:
            self.tsdf_maps = None

    def _indices_to_binary_labels(self, min_indices, max_indices):
        """Convert min/max indices back to binary labels for TSDF computation."""
        binary_labels = np.zeros((len(min_indices), 360))
        
        for i, (min_idx, max_idx) in enumerate(zip(min_indices, max_indices)):
            if min_idx <= max_idx:
                # Normal case: no wrapping
                binary_labels[i, min_idx:max_idx+1] = 1
            else:
                # Circular case: wrapping around
                binary_labels[i, min_idx:] = 1  # From min_idx to end
                binary_labels[i, :max_idx+1] = 1  # From start to max_idx
        
        return binary_labels

    def _precompute_tsdf_maps(self):
        """Pre-compute TSDF maps for all samples."""
        # Use the binary labels for obstacle detection
        obstacle_predictions = (self.labels > 0.5).astype(np.float32)
        tsdf_maps = lidar_to_tsdf_obstacles(self.data, obstacle_predictions)
        return tsdf_maps

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        distances = torch.tensor(self.data[idx], dtype=torch.float32)
        
        # Return the min/max indices as the target for training
        min_idx = torch.tensor(self.min_indices[idx], dtype=torch.long)
        max_idx = torch.tensor(self.max_indices[idx], dtype=torch.long)
        
        mask = torch.tensor(self.mask[idx], dtype=torch.float32)
        
        if self.use_tsdf and self.tsdf_maps is not None:
            tsdf = torch.tensor(self.tsdf_maps[idx], dtype=torch.float32)
            return distances, (min_idx, max_idx), mask, tsdf
        else:
            return distances, (min_idx, max_idx), mask

# Enhanced ResNet with TSDF integration
class ResNetWithTSDF(nn.Module):
    def __init__(self, use_tsdf=True):
        super().__init__()
        self.use_tsdf = use_tsdf
        
        # LIDAR processing branch
        self.lidar_layer1 = ResidualBlock(1, 16)
        self.lidar_layer2 = ResidualBlock(16, 32)
        self.lidar_layer3 = ResidualBlock(32, 64)
        
        if use_tsdf:
            # TSDF processing branch
            self.tsdf_conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
            self.tsdf_conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
            self.tsdf_conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            
            # Feature fusion
            self.fusion_conv = nn.Conv1d(64 + 64, 64, kernel_size=1)
        
        # Output heads for min and max indices
        self.min_classifier = nn.Conv1d(64, 360, kernel_size=1)  # Predict min index
        self.max_classifier = nn.Conv1d(64, 360, kernel_size=1)  # Predict max index
        
    def forward(self, lidar_data, tsdf_data=None):
        # Process LIDAR data
        x_lidar = self.lidar_layer1(lidar_data)
        x_lidar = self.lidar_layer2(x_lidar)
        x_lidar = self.lidar_layer3(x_lidar)
        
        if self.use_tsdf and tsdf_data is not None:
            # Process TSDF data
            x_tsdf = torch.relu(self.tsdf_conv1(tsdf_data))
            x_tsdf = torch.relu(self.tsdf_conv2(x_tsdf))
            x_tsdf = torch.relu(self.tsdf_conv3(x_tsdf))  # (B, 64, H, W)
            
            # Flatten spatial dimensions and use a linear layer to get 360 features
            b, c, h, w = x_tsdf.shape
            x_tsdf_flat = x_tsdf.view(b, c, -1)  # (B, 64, H*W)
            # Project to 360 features using a linear layer
            if not hasattr(self, 'tsdf_proj'):
                self.tsdf_proj = nn.Linear(h * w, 360).to(x_tsdf.device)
            x_tsdf_proj = self.tsdf_proj(x_tsdf_flat)  # (B, 64, 360)
            
            # Fuse features
            x_fused = torch.cat([x_lidar, x_tsdf_proj], dim=1)  # (B, 128, 360)
            x_fused = self.fusion_conv(x_fused)  # (B, 64, 360)
            
            # Predict min and max indices
            min_logits = self.min_classifier(x_fused)  # (B, 360, 360)
            max_logits = self.max_classifier(x_fused)  # (B, 360, 360)
        else:
            # Predict min and max indices
            min_logits = self.min_classifier(x_lidar)  # (B, 360, 360)
            max_logits = self.max_classifier(x_lidar)  # (B, 360, 360)
        
        # Average pool to get single predictions
        min_pred = torch.mean(min_logits, dim=2)  # (B, 360)
        max_pred = torch.mean(max_logits, dim=2)  # (B, 360)
        
        return min_pred, max_pred

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

# Load the datasets (keeping your original data loading)
# Set path to be data directory
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # pseudoslam/envs (data/ lives there)
data_dir = os.path.join(base_dir, "data")
# Load training and testing data (LIDAR and labels)
training_lidar = np.load(os.path.join(data_dir, "training_lidar.npy")) 
training_labels = np.load(os.path.join(data_dir, "training_labels.npy")) 
testing_lidar = np.load(os.path.join(data_dir, "testing_lidar.npy")) 
labels_test = np.load(os.path.join(data_dir, "testing_labels.npy"))  

def labels_to_min_max(labels):
    min_indices = []
    max_indices = []
    
    for i, row in enumerate(labels):
        ones_indices = np.where(row == 1)[0]
        
        if len(ones_indices) == 0:
            # No 1s found
            min_indices.append(0)
            max_indices.append(row.shape[0] - 1)
            continue
        
        if len(ones_indices) == 1:
            # Only one 1 found
            min_indices.append(ones_indices[0])
            max_indices.append(ones_indices[0])
            continue
        
        # Check if the 1s wrap around (circular range)
        # Look for the largest gap between consecutive 1s
        sorted_indices = np.sort(ones_indices)
        gaps = np.diff(sorted_indices)
        
        # Also check the wrap-around gap
        wrap_gap = (row.shape[0] - sorted_indices[-1]) + sorted_indices[0]
        
        # Find the largest gap
        max_gap_idx = np.argmax(gaps)
        max_gap = gaps[max_gap_idx]
        
        if wrap_gap > max_gap:
            # The largest gap is the wrap-around gap
            # So the range wraps around: min is after the gap, max is before it
            min_indices.append(sorted_indices[-1])  # Last index before wrap
            max_indices.append(sorted_indices[0])   # First index after wrap
        else:
            # Normal case: range doesn't wrap around
            min_indices.append(sorted_indices[0])
            max_indices.append(sorted_indices[-1])
    
    min_indices = np.array(min_indices)
    max_indices = np.array(max_indices)
    
    return min_indices, max_indices


# Change labels to min-max indices using labels_to_min_max function
training_min_indices, training_max_indices = labels_to_min_max(training_labels)
testing_min_indices, testing_max_indices = labels_to_min_max(labels_test)


# Masking (keeping your original masking)
training_mask = np.random.binomial(1, 0.9, training_labels.shape).astype(np.float32)
testing_mask = np.ones_like(labels_test, dtype=np.float32)  # all valid in testing

# Create datasets and dataloaders (now with TSDF support)
train_dataset = LidarTSDFDataset(training_lidar, (training_min_indices, training_max_indices), training_mask, use_tsdf=True)
test_dataset = LidarTSDFDataset(testing_lidar, (testing_min_indices, testing_max_indices), testing_mask, use_tsdf=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Create model (enhanced with TSDF)
resnet = ResNetWithTSDF(use_tsdf=True).to(device)

def min_max_loss(min_pred, max_pred, min_target, max_target):
    """
    Custom loss function for min-max index prediction.
    Uses sigmoid focal loss for both min and max predictions.
    """
    # Convert targets to one-hot encoding
    min_target_one_hot = F.one_hot(min_target, num_classes=360).float()
    max_target_one_hot = F.one_hot(max_target, num_classes=360).float()
    
    # Compute focal loss for min and max predictions
    min_loss = sigmoid_focal_loss(min_pred, min_target_one_hot, reduction='mean')
    max_loss = sigmoid_focal_loss(max_pred, max_target_one_hot, reduction='mean')
    
    return min_loss + max_loss

optimizer = optim.Adam(resnet.parameters(), lr=learning_rate, weight_decay=1e-4)

# Training loop (keeping your original structure)
def train_model():
    n_total_steps = len(train_loader)
    loss_history = []
    
    for epoch in range(num_epochs):
        resnet.train()
        epoch_loss = 0.0
        
        for i, batch_data in enumerate(train_loader):
            if len(batch_data) == 4:  # With TSDF
                distances, labels, mask, tsdf = batch_data
                distances = distances.unsqueeze(1).to(device)  # (B, 1, 360)
                min_target, max_target = labels
                min_target = min_target.to(device)  # (B,)
                max_target = max_target.to(device)  # (B,)
                mask = mask.to(device)  # (B, 360)
                tsdf = tsdf.unsqueeze(1).to(device)  # (B, 1, 128, 128)
                
                min_pred, max_pred = resnet(distances, tsdf)  # (B, 360), (B, 360)
            else:  # Without TSDF (fallback)
                distances, labels, mask = batch_data
                distances = distances.unsqueeze(1).to(device)  # (B, 1, 360)
                min_target, max_target = labels
                min_target = min_target.to(device)  # (B,)
                max_target = max_target.to(device)  # (B,)
                mask = mask.to(device)  # (B, 360)
                
                min_pred, max_pred = resnet(distances)  # (B, 360), (B, 360)
            
            # Compute loss
            loss = min_max_loss(min_pred, max_pred, min_target, max_target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')
        
        loss_history.append(epoch_loss / n_total_steps)
    
    return loss_history

print('Finished training')

# Testing (keeping your original testing structure)
n_correct = 0

# Updated testing loop
def test_model():
    resnet.eval()
    with torch.no_grad():
        total_percent_error = 0.0
        total_samples = 0
        
        for batch_data in test_loader:
            if len(batch_data) == 4:  # With TSDF
                distances, labels, mask, tsdf = batch_data
                distances = distances.unsqueeze(1).to(device)
                min_target, max_target = labels
                min_target = min_target.to(device)
                max_target = max_target.to(device)
                tsdf = tsdf.unsqueeze(1).to(device)
                min_pred, max_pred = resnet(distances, tsdf)
            else:  # Without TSDF (fallback)
                distances, labels, mask = batch_data
                distances = distances.unsqueeze(1).to(device)
                min_target, max_target = labels
                min_target = min_target.to(device)
                max_target = max_target.to(device)
                min_pred, max_pred = resnet(distances)
            
            # Get predicted indices
            min_predicted = torch.argmax(min_pred, dim=1)
            max_predicted = torch.argmax(max_pred, dim=1)
            
            # Calculate target distance (ground truth)
            distance_target = (max_target - min_target + 360) % 360
            
            # Calculate predicted distance
            distance_predicted = (max_predicted - min_predicted + 360) % 360
            
            # Calculate absolute percentage error for each sample
            # Add small epsilon to avoid division by zero
            epsilon = 1e-8
            abs_percent_error = torch.abs(distance_predicted - distance_target) / (distance_target + epsilon)
            
            # Accumulate errors
            total_percent_error += abs_percent_error.sum().item()
            total_samples += min_target.size(0)

        # Calculate average percentage error and convert to accuracy
        avg_percent_error = total_percent_error / total_samples
        accuracy = (1 - avg_percent_error) * 100.0
        
        print(f'Test Accuracy: {accuracy:.2f}%')
        print(f'Average Percentage Error: {avg_percent_error:.4f}')

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

# Train in main
if __name__ == "__main__":
    loss_history = train_model()
    min_accuracy, max_accuracy = test_model()

# Save the trained model (keeping your original save path)
model_path = os.path.join(data_dir, "resnet_lidar_model_tsdf.pth")
torch.save(resnet.state_dict(), model_path)
print(f'Model saved to {model_path}')