import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import numpy as np
import os
import torch.nn.functional as F
import matplotlib.pyplot as plt


# Function where you input an index and it shows you the lidar, tsdf, and label as printed array
def print_sample_data(lidar_data, label, index=0):
    tsdf_data = lidar_to_tsdf_obstacles(lidar_data[index:index+1], (label[index:index+1] > 0.5).astype(np.float32))
    print("LiDAR Data:")
    print(lidar_data[index])
    print("\nTSDF Data:")
    # np.set_printoptions(threshold=np.inf)  # Print full array without truncation
    # print full tsdf data (everything)
    print(tsdf_data)
    print("\nLabel:")
    print(label[index])

def visualize_input_sample(lidar_data, tsdf_data, label, pred, mask=None):
    fig, axs = plt.subplots(4, 1, figsize=(16, 8))
    
    axs[0].plot(lidar_data[0].cpu().numpy(), label='LiDAR')
    axs[0].legend()
    
    axs[1].imshow(tsdf_data[0].cpu().numpy().sum(axis=0), aspect='auto')
    axs[1].set_title("TSDF sum")
    
    axs[2].plot(label[0].cpu().numpy(), label='Ground Truth')
    axs[2].plot(pred[0].detach().cpu().numpy(), label='Prediction')
    axs[2].legend()
    
    if mask is not None:
        axs[3].plot(mask[0].cpu().numpy(), label='Valid Mask')
        axs[3].legend()
        
    plt.tight_layout()
    plt.show()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
num_epochs = 50
batch_size = 64
learning_rate = 0.001

torch.autograd.set_detect_anomaly(True)
torch.cuda.empty_cache()


def lidar_to_tsdf_obstacles(lidar_data, obstacle_predictions, img_dim=32, max_range=200.0, truncation=3.0):
    """
    Converts LiDAR polar data and obstacle labels into a 2-channel TSDF:
    - Channel 0: positive SDF (free space)
    - Channel 1: negative SDF (obstacle regions)

    Args:
        lidar_data: (B, 360) LiDAR distance readings in meters
        obstacle_predictions: (B, 360) binary labels (1 = obstacle)
        img_dim: output TSDF resolution (e.g., 64 → 64x64 image)
        max_range: maximum LiDAR range in meters
        truncation: truncation distance in meters for SDF

    Returns:
        tsdf_maps: (B, 2, img_dim, img_dim) TSDF tensor
    """

    # Ensure batch dimension
    if lidar_data.ndim == 1:
        lidar_data = np.expand_dims(lidar_data, axis=0)
    if obstacle_predictions.ndim == 1:
        obstacle_predictions = np.expand_dims(obstacle_predictions, axis=0)

    B = lidar_data.shape[0]
    tsdf_maps = np.zeros((B, 2, img_dim, img_dim), dtype=np.float32)

    center = img_dim // 2
    angles = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    x_coords, y_coords = np.meshgrid(np.arange(img_dim), np.arange(img_dim))
    pix_per_meter = img_dim / (2 * max_range)

    for b in range(B):
        for i in range(360):
            distance = lidar_data[b, i]
            is_obstacle = int(obstacle_predictions[b, i])

            if distance <= 0 or distance >= max_range:
                continue

            # Convert polar to grid coordinates
            x = distance * np.cos(angles[i])
            y = distance * np.sin(angles[i])
            xg = int(center + x * pix_per_meter)
            yg = int(center + y * pix_per_meter)

            if not (0 <= xg < img_dim and 0 <= yg < img_dim):
                continue

            # Compute per-pixel Euclidean distance from (xg, yg)
            dist_map = np.sqrt((x_coords - xg) ** 2 + (y_coords - yg) ** 2)
            mask = dist_map < truncation * pix_per_meter
            if not np.any(mask):
                continue

            # Convert distance to real-world meters
            sdf = dist_map * (max_range / (img_dim / 2))  # pixels → meters
            sdf = np.clip(sdf, 0, truncation)
            signed_sdf = sdf / truncation

            if is_obstacle:
                # Write to channel 1 (obstacle): negative SDF
                tsdf_maps[b, 1][mask] = np.minimum(tsdf_maps[b, 1][mask], -signed_sdf[mask])
            else:
                # Write to channel 0 (free space): positive SDF
                tsdf_maps[b, 0][mask] = np.maximum(tsdf_maps[b, 0][mask], signed_sdf[mask])

    return tsdf_maps


def visualize_tsdf(tsdf_map):
    fig, axs = plt.subplots(1, 1, figsize=(5, 5))
    # tsdf_map shape: (B, 1, H, W), visualize first sample and only channel
    axs.imshow(tsdf_map[0, 0], cmap='viridis')
    axs.set_title("TSDF Map")
    plt.tight_layout()
    plt.show()

def labels_to_min_max(labels):
    """
    For each label row, find all contiguous ranges of 1s (with wraparound).
    Returns two lists:
      - min_indices: list of lists of min indices for each range in each row
      - max_indices: list of lists of max indices for each range in each row
    """
    labels = np.atleast_2d(labels)
    min_indices_all, max_indices_all = [], []
    for row in labels:
        row = np.asarray(row)
        n = row.shape[0]
        # Find all indices where value is 1
        ones = np.where(row == 1)[0]
        if len(ones) == 0:
            min_indices_all.append([0])
            max_indices_all.append([n - 1])
            continue
        # Find breaks between contiguous regions
        diffs = np.diff(np.concatenate(([ones[-1] - n], ones)))
        breaks = np.where(diffs != 1)[0]
        # Split into contiguous regions
        ranges = np.split(ones, breaks[1:])
        min_indices, max_indices = [], []
        for r in ranges:
            if len(r) == 0:
                continue
            min_indices.append(r[0])
            max_indices.append(r[-1])
        # Handle wraparound: if first and last region are contiguous
        if len(ranges) > 1 and (ranges[0][0] == 0 and ranges[-1][-1] == n - 1):
            # Merge first and last
            merged = np.concatenate((ranges[-1], ranges[0]))
            min_indices = [merged[0]] + min_indices[1:-1]
            max_indices = max_indices[1:-1] + [merged[-1]]
        min_indices_all.append(min_indices)
        max_indices_all.append(max_indices)
    return min_indices_all, max_indices_all

def generate_multi_obstacle_targets(labels):
    """
    Generate targets that can handle multiple obstacles per sample.
    Returns: list of lists of (min_idx, max_idx) tuples for each sample.
    Handles wraparound and multiple regions.
    """
    batch_targets = []
    labels = labels.cpu().numpy() if hasattr(labels, "cpu") else labels
    for label_row in labels:
        label_row = np.asarray(label_row)
        n = label_row.shape[0]
        # Pad for diff
        padded = np.concatenate([[0], label_row, [0]])
        diff = np.diff(padded)
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0] - 1

        # Handle wraparound: if first and last are 1, and regions at both ends
        if label_row[0] == 1 and label_row[-1] == 1:
            if len(starts) > 1:
                # Merge first and last
                starts = np.concatenate(([starts[-1]], starts[:-1]))
                ends = np.concatenate((ends[1:], [ends[0]]))
            # If only one region, it's already correct

        obstacle_regions = []
        for s, e in zip(starts, ends):
            obstacle_regions.append([int(s), int(e)])
        batch_targets.append(obstacle_regions)
    return batch_targets

def multi_obstacle_loss(min_pred_logits, max_pred_logits, obstacle_targets, gating_info, alpha=0.5, beta=0.1):
    """
    Loss function that handles multiple obstacles per sample
    """
    # print(f'Obstacle target shape: {len(obstacle_targets)}')
    batch_size = min_pred_logits.shape[0]
    total_loss = 0.0
    total_samples = len(obstacle_targets)
    
    components = {'standard_loss': 0.0, 'confidence_loss': 0.0, 'gate_reg': 0.0, 'uncertainty_reg': 0.0}
    
    for batch_idx in range(batch_size):
        obstacle_regions = obstacle_targets[batch_idx]
        
        # For multiple obstacles, we can use different strategies:
        # Strategy 1: Use the largest obstacle region
        largest_region = max(obstacle_regions, key=lambda x: (x[1] - x[0] + 360) % 360)
        min_target = torch.tensor(largest_region[0], device=min_pred_logits.device)
        max_target = torch.tensor(largest_region[1], device=min_pred_logits.device)

        # Instead of MSE treat as classification for min and max (based on probababilities) use CE
        position_loss = F.cross_entropy(min_pred_logits[batch_idx:batch_idx+1], min_target.unsqueeze(0)) + \
                       F.cross_entropy(max_pred_logits[batch_idx:batch_idx+1], max_target.unsqueeze(0))

        sample_loss = alpha * position_loss + (1-alpha) # * ce_loss
        total_loss += sample_loss
        components['standard_loss'] += sample_loss.item()
    
    if total_samples == 0:
        return torch.tensor(0.0, device=min_pred_logits.device), components
    
    # Average the loss
    total_loss = total_loss / total_samples
    components['standard_loss'] /= total_samples
    
    # Add gating regularization
    min_gate = gating_info['min_gate']
    max_gate = gating_info['max_gate']
    uncertainty = gating_info['uncertainty']
    confidence = gating_info['confidence']
    
    confidence_loss = F.mse_loss(confidence.mean(), torch.ones_like(confidence.mean()) * 0.8)
    gate_reg = torch.mean(torch.abs(min_gate - 0.5)) + torch.mean(torch.abs(max_gate - 0.5))
    uncertainty_reg = F.mse_loss(uncertainty * confidence, torch.zeros_like(uncertainty * confidence))
    
    components['confidence_loss'] = confidence_loss.item()
    components['gate_reg'] = gate_reg.item()
    components['uncertainty_reg'] = uncertainty_reg.item()
    
    total_loss = total_loss + beta * (confidence_loss + gate_reg + uncertainty_reg)
    
    return total_loss, components


class LidarTSDFDataset(Dataset):
    def __init__(self, data, labels, mask=None, tsdf_cache_path=None):
        self.data = data
        self.mask = mask if mask is not None else np.ones((len(data), 360))

        if isinstance(labels, tuple):
            self.min_indices, self.max_indices = labels
            binary_labels = np.zeros((len(self.min_indices), 360))
            for i, (min_idx_list, max_idx_list) in enumerate(zip(self.min_indices, self.max_indices)):
                for min_idx, max_idx in zip(min_idx_list, max_idx_list):
                    if min_idx <= max_idx:
                        binary_labels[i, min_idx:max_idx+1] = 1
                    else:
                        binary_labels[i, min_idx:] = 1
                        binary_labels[i, :max_idx+1] = 1
            self.labels = binary_labels
        else:
            self.labels = labels
            self.min_indices, self.max_indices = labels_to_min_max(labels)

        min_len = min(len(self.data), len(self.labels))
        if len(self.data) != len(self.labels):
            print(f"Warning: data length ({len(self.data)}) != labels length ({len(self.labels)})")
            print(f"Truncating to minimum length: {min_len}")
            self.data = self.data[:min_len]
            self.labels = self.labels[:min_len]
            self.min_indices = self.min_indices[:min_len]
            self.max_indices = self.max_indices[:min_len]
            if self.mask is not None:
                self.mask = self.mask[:min_len]

        # === TSDF Caching ===
        if tsdf_cache_path and os.path.exists(tsdf_cache_path):
            print(f"Loading TSDF from {tsdf_cache_path}")
            self.tsdf_maps = np.load(tsdf_cache_path)
        else:
            print("Computing TSDF maps...")
            self.tsdf_maps = lidar_to_tsdf_obstacles(self.data, (self.labels > 0.5).astype(np.float32))
            if tsdf_cache_path:
                np.save(tsdf_cache_path, self.tsdf_maps)
                print(f"Saved TSDF maps to {tsdf_cache_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Raw data
        lidar_data = self.data[idx]
        label = self.labels[idx]
        
        # Compute TSDF (returns shape (1, 1, H, W))
        tsdf_np = lidar_to_tsdf_obstacles(np.expand_dims(lidar_data, axis=0),
                                        np.expand_dims(label, axis=0))  # (1, 1, 64, 64)

        tsdf_tensor = torch.from_numpy(tsdf_np[0]).float()  # (1, 64, 64)

        # Tensors
        distances = torch.tensor(lidar_data, dtype=torch.float32).unsqueeze(0)  # (1, 360)
        label_tensor = torch.tensor(label, dtype=torch.long)                    # (360,)
        mask_tensor = torch.tensor(self.mask[idx], dtype=torch.float32)        # (360,)

        return distances, label_tensor, mask_tensor, tsdf_tensor
    
    def visualize_sample(self, lidar, tsdf, label, mask):
        lidar = lidar.cpu().numpy()
        tsdf = tsdf.cpu().numpy()
        label = label.cpu().numpy()
        mask = mask.cpu().numpy()

        fig, axs = plt.subplots(4, 1, figsize=(12, 8))
        axs[0].plot(lidar[0])
        axs[0].set_title("LiDAR")

        axs[1].imshow(tsdf[0].sum(axis=0))
        axs[1].set_title("TSDF Sum")

        axs[2].plot(label[0], label='Label')
        axs[2].legend()

        axs[3].plot(mask[0], label='Mask')
        axs[3].legend()

        plt.tight_layout()
        plt.show()


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

class GatingModule(nn.Module):
    """
    Improved gating module that learns to weight min and max predictions based on input features,
    with better context aggregation and more stable gating/uncertainty outputs.
    """
    def __init__(self, feature_dim=64, hidden_dim=32):
        super().__init__()
        self.feature_dim = feature_dim

        # Global context: squeeze spatial, then expand
        self.global_fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(feature_dim, hidden_dim, 1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, 1),
            nn.ReLU()
        )

        # Local context: keep spatial
        self.local_fc = nn.Sequential(
            nn.Conv1d(feature_dim, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU()
        )

        # Gating/uncertainty head: use LayerNorm for stability
        self.gate_head = nn.Sequential(
            nn.Conv1d(hidden_dim * 2, hidden_dim, 1),
            nn.LayerNorm([hidden_dim, 360]),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, 3, 1),  # min_gate, max_gate, uncertainty
            nn.Sigmoid()
        )

        # Confidence head: use LayerNorm for stability
        self.conf_head = nn.Sequential(
            nn.Conv1d(hidden_dim * 2, hidden_dim, 1),
            nn.LayerNorm([hidden_dim, 360]),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, features):
        batch, channels, seq_len = features.shape

        # Global context: (B, hidden_dim, 1) -> (B, hidden_dim, seq_len)
        global_ctx = self.global_fc(features)
        global_ctx = global_ctx.expand(-1, -1, seq_len)

        # Local context: (B, hidden_dim, seq_len)
        local_ctx = self.local_fc(features)

        # Combine
        ctx = torch.cat([global_ctx, local_ctx], dim=1)  # (B, hidden_dim*2, seq_len)

        # Gating/uncertainty
        gates = self.gate_head(ctx)  # (B, 3, seq_len)
        min_gate = gates[:, 0:1, :]
        max_gate = gates[:, 1:2, :]
        uncertainty = gates[:, 2:3, :]

        # Confidence
        confidence = self.conf_head(ctx)  # (B, 1, seq_len)

        # Softmax over gates for better normalization
        gate_cat = torch.cat([min_gate, max_gate], dim=1)  # (B, 2, seq_len)
        gate_soft = F.softmax(gate_cat, dim=1)
        min_gate = gate_soft[:, 0:1, :]
        max_gate = gate_soft[:, 1:2, :]

        # Clamp uncertainty for numerical stability
        uncertainty = torch.clamp(uncertainty, min=0.01, max=0.99)

        return min_gate, max_gate, uncertainty, confidence

class ImprovedResNetWithGating(nn.Module):
    def __init__(self):
        super().__init__()
        # LiDAR processing
        self.lidar_layer1 = ResidualBlock(1, 16)
        self.lidar_layer2 = ResidualBlock(16, 32)
        self.lidar_layer3 = ResidualBlock(32, 64)
        
        # TSDF Conv1D path
        self.tsdf_conv1d = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2)
        )

        # Fusion layers (should match dimensions from LiDAR+TSDF concat)
        self.fusion_conv1 = nn.Conv1d(192, 128, kernel_size=3, padding=1)
        self.fusion_conv2 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
        
        # Gating module
        self.gating_module = GatingModule(feature_dim=64, hidden_dim=32)
        
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
        
        # Gated output head
        self.gated_output = nn.Sequential(
            nn.Conv1d(64, 32, 1),
            nn.ReLU(),
            nn.Conv1d(32, 2, 1)  # 2 outputs: min and max
        )
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, lidar_data, tsdf_data):
        # Process LiDAR
        x_lidar = self.lidar_layer1(lidar_data)  # (B, 16, 360)
        x_lidar = self.lidar_layer2(x_lidar)     # (B, 32, 360)
        x_lidar = self.lidar_layer3(x_lidar)     # (B, 64, 360)

        # TSDF shape check and interpolation
        while tsdf_data.ndim > 4:
            tsdf_data = tsdf_data.squeeze(0)
        assert tsdf_data.ndim == 4, f"Expected 4D TSDF, got {tsdf_data.shape}"

        # TSDF should be (B, 2, H, 360) → average over H (radial depth)
        if tsdf_data.shape[-1] != 360:
            tsdf_data = F.interpolate(tsdf_data, size=(tsdf_data.shape[2], 360), mode='bilinear', align_corners=False)
        tsdf_radial = tsdf_data.mean(dim=2)  # (B, 2, 360)
        assert tsdf_radial.shape[1] == 2, f"TSDF channels should be 2, got {tsdf_radial.shape[1]}"

        # TSDF Conv1D
        tsdf_feat = self.tsdf_conv1d(tsdf_radial)  # (B, 128, 360)

        # Fuse LiDAR and TSDF features
        x_fused = torch.cat([x_lidar, tsdf_feat], dim=1)  # (B, 192, 360)
        x_fused = torch.relu(self.fusion_conv1(x_fused))  # (B, 128, 360)
        x_fused = torch.relu(self.fusion_conv2(x_fused))  # (B, 64, 360)
        x_fused = self.dropout(x_fused)

        # Gating variables
        min_gate, max_gate, uncertainty, confidence = self.gating_module(x_fused)

        # Separate predictions
        min_logits = self.min_head(x_fused).squeeze(1)  # (B, 360)
        max_logits = self.max_head(x_fused).squeeze(1)  # (B, 360)

        # Gated predictions
        gated_output = self.gated_output(x_fused)        # (B, 2, 360)
        gated_min_logits = gated_output[:, 0, :]         # (B, 360)
        gated_max_logits = gated_output[:, 1, :]         # (B, 360)

        # Apply gating: blend raw and gated logits
        final_min_logits = min_gate.squeeze(1) * min_logits + (1 - min_gate.squeeze(1)) * gated_min_logits
        final_max_logits = max_gate.squeeze(1) * max_logits + (1 - max_gate.squeeze(1)) * gated_max_logits

        # Apply uncertainty weighting
        final_min_logits = final_min_logits * (1 - uncertainty.squeeze(1)) + uncertainty.squeeze(1) * gated_min_logits
        final_max_logits = final_max_logits * (1 - uncertainty.squeeze(1)) + uncertainty.squeeze(1) * gated_max_logits

        return final_min_logits, final_max_logits, {
            'min_gate': min_gate,
            'max_gate': max_gate,
            'uncertainty': uncertainty,
            'confidence': confidence,
            'raw_min_logits': min_logits,
            'raw_max_logits': max_logits
        }
    
def debug_visualize_batch(lidar, tsdf, label, pred, mask=None):
    lidar = lidar.detach().cpu().numpy()
    tsdf = tsdf.detach().cpu().numpy()
    label = label.detach().cpu().numpy()
    pred = pred.detach().cpu().numpy()
    mask = mask.detach().cpu().numpy() if mask is not None else None

    fig, axs = plt.subplots(4, 1, figsize=(16, 8))
    axs[0].plot(lidar[0][0], label='LiDAR')
    axs[0].legend()

    axs[1].imshow(tsdf[0].sum(axis=0), aspect='auto')
    axs[1].set_title("TSDF Sum")

    axs[2].plot(label[0], label='Ground Truth')
    axs[2].plot(pred[0], label='Prediction')
    axs[2].legend()

    if mask is not None:
        axs[3].plot(mask[0], label='Mask')
        axs[3].legend()

    plt.tight_layout()
    plt.show()


def circular_distance(a, b, max_val=360):
    """Calculate circular distance between two angles"""
    diff = torch.abs(a - b)
    return torch.minimum(diff, max_val - diff)

def calculate_span(min_idx, max_idx):
    """Calculate proper span handling wraparound"""
    span = torch.where(min_idx <= max_idx, 
                      max_idx - min_idx + 1,
                      360 - min_idx + max_idx + 1)
    return span

# Load data
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, "data")
training_lidar = np.load(os.path.join(data_dir, "new_training_lidar.npy"))
training_labels = np.load(os.path.join(data_dir, "new_training_labels.npy"))
testing_lidar = np.load(os.path.join(data_dir, "new_testing_lidar.npy"))
labels_test = np.load(os.path.join(data_dir, "new_testing_labels.npy"))

# # Visualize the 10000th sample (index 10000)
# idx = 10000
# training_tsdf = lidar_to_tsdf_obstacles(training_lidar[idx:idx+1], (training_labels[idx:idx+1] > 0.5).astype(np.float32))
# # For visualization, use idx for lidar and labels, and index 0 for TSDF (since it's only one sample)
# visualize_input_sample(training_lidar, training_tsdf, training_labels, index=idx)
# visualize_input_sample(training_lidar[idx:idx+1], training_tsdf, training_labels[idx:idx+1], index=0)
# # Alternatively, to avoid confusion, visualize TSDF with index 0:
# # visualize_input_sample(training_lidar, training_tsdf, training_labels, index=0)


# Debug: Print shapes to verify
print(f"Training lidar shape: {training_lidar.shape}")
print(f"Training labels shape: {training_labels.shape}")
print(f"Testing lidar shape: {testing_lidar.shape}")
print(f"Testing labels shape: {labels_test.shape}")

training_min_indices, training_max_indices = labels_to_min_max(training_labels)
testing_min_indices, testing_max_indices = labels_to_min_max(labels_test)

print(f"Training min indices length: {len(training_min_indices)}")
print(f"Training max indices length: {len(training_max_indices)}")
print(f"Testing min indices length: {len(testing_min_indices)}")
print(f"Testing max indices length: {len(testing_max_indices)}")

training_mask = np.where(training_labels == 1, 1, np.random.binomial(1, 0.9, training_labels.shape))
training_mask = training_mask.astype(np.float32)
testing_mask = np.ones_like(labels_test, dtype=np.float32)

train_dataset = LidarTSDFDataset(
    training_lidar, 
    (training_min_indices, training_max_indices), 
    training_mask,
    tsdf_cache_path=os.path.join(data_dir, "tsdf_train.npy")
)

test_dataset = LidarTSDFDataset(
    testing_lidar, 
    (testing_min_indices, testing_max_indices), 
    testing_mask,
    tsdf_cache_path=os.path.join(data_dir, "tsdf_test.npy")
)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)


# Use improved model with gating
resnet = ImprovedResNetWithGating().to(device)

def train_model():
    optimizer = optim.Adam(resnet.parameters(), lr=0.0005, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    loss_history = []
    loss_components = {'standard': [], 'confidence': [], 'gate_reg': [], 'uncertainty_reg': []}
    
    for epoch in range(num_epochs):
        resnet.train()
        epoch_loss = 0.0
        epoch_components = {'standard': 0.0, 'confidence': 0.0, 'gate_reg': 0.0, 'uncertainty_reg': 0.0}
        
        for batch_idx, (distances, labels, _, tsdf) in enumerate(train_loader):

            distances = distances.to(device)
            tsdf = tsdf.to(device)
            
            # Generate multi-obstacle targets
            obstacle_targets = generate_multi_obstacle_targets(labels)
            
            optimizer.zero_grad()
            min_pred, max_pred, gating_info = resnet(distances, tsdf)

            # Use multi-obstacle loss
            loss, components = multi_obstacle_loss(min_pred, max_pred, obstacle_targets, gating_info)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(resnet.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            for key, value in components.items():
                key_map = {
                    'standard_loss': 'standard',
                    'confidence_loss': 'confidence',
                    'gate_reg': 'gate_reg',
                    'uncertainty_reg': 'uncertainty_reg'
                }
                if key in key_map:
                    epoch_components[key_map[key]] += value
            
            if batch_idx % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], '
                      f'Loss: {loss.item():.4f}')
        
        scheduler.step()
        avg_epoch_loss = epoch_loss / len(train_loader)
        loss_history.append(avg_epoch_loss)
        
        for key in epoch_components:
            loss_components[key].append(epoch_components[key] / len(train_loader))
        
        print(f'Epoch [{epoch+1}/{num_epochs}] Average Loss: {avg_epoch_loss:.4f}')
    
    return loss_history, loss_components

def test_model_with_gating(loss_history, loss_components):
    resnet.eval()
    total_samples = 0
    correct_predictions = 0
    total_absolute_error = 0.0
    
    gate_stats = {'min_gate_avg': 0.0, 'max_gate_avg': 0.0, 'uncertainty_avg': 0.0, 'confidence_avg': 0.0}
    
    for distances, labels, _, tsdf in test_loader:
        distances = distances.to(device)
        # For classification, get min and max indices from binary label
        min_target = torch.argmax(labels, dim=1).to(device)
        max_target = (labels.size(1) - 1 - torch.argmax(torch.flip(labels, dims=[1]), dim=1)).to(device)
        tsdf = tsdf.to(device)
        
        min_pred, max_pred, gating_info = resnet(distances, tsdf)
        min_predicted = torch.argmax(min_pred, dim=1)
        max_predicted = torch.argmax(max_pred, dim=1)
        
        # Calculate target and predicted distances
        distance_target = (max_target - min_target + 360) % 360
        distance_predicted = (max_predicted - min_predicted + 360) % 360
        
        # Calculate absolute error
        absolute_error = torch.abs(distance_predicted - distance_target)
        total_absolute_error += absolute_error.sum().item()
        
        # Count correct predictions within tolerance
        tolerance = 10  # degrees
        correct_predictions += (absolute_error <= tolerance).sum().item()
        
        total_samples += min_target.size(0)
        
        # Collect gating statistics
        gate_stats['min_gate_avg'] += gating_info['min_gate'].mean().item()
        gate_stats['max_gate_avg'] += gating_info['max_gate'].mean().item()
        gate_stats['uncertainty_avg'] += gating_info['uncertainty'].mean().item()
        gate_stats['confidence_avg'] += gating_info['confidence'].mean().item()
        gate_stats['confidence_avg'] += gating_info['confidence'].mean().item()

    # Calculate metrics
    avg_absolute_error = total_absolute_error / total_samples
    tolerance_accuracy = (correct_predictions / total_samples) * 100.0
    
    # Average gating statistics
    num_batches = len(test_loader)
    for key in gate_stats:
        gate_stats[key] /= num_batches
    
    print(f'Tolerance Accuracy (±{tolerance}°): {tolerance_accuracy:.2f}%')
    print(f'Average Absolute Error: {avg_absolute_error:.2f} degrees')
    print(f'Gating Statistics:')
    print(f'  Min Gate Average: {gate_stats["min_gate_avg"]:.3f}')
    print(f'  Max Gate Average: {gate_stats["max_gate_avg"]:.3f}')
    print(f'  Uncertainty Average: {gate_stats["uncertainty_avg"]:.3f}')
    print(f'  Confidence Average: {gate_stats["confidence_avg"]:.3f}')
    
    # Plot results
    plt.figure(figsize=(15, 10))
    
    # Loss history
    plt.subplot(2, 3, 1)
    plt.plot(range(num_epochs), loss_history, label='Total Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    
    # Loss components
    plt.subplot(2, 3, 2)
    for key, values in loss_components.items():
        plt.plot(range(num_epochs), values, label=key)
    plt.xlabel('Epochs')
    plt.ylabel('Loss Component')
    plt.title('Loss Components')
    plt.legend()
    
    # Gating statistics
    plt.subplot(2, 3, 3)
    gate_names = list(gate_stats.keys())
    gate_values = list(gate_stats.values())
    plt.bar(gate_names, gate_values)
    plt.title('Gating Statistics')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    return tolerance_accuracy, avg_absolute_error, gate_stats

if __name__ == "__main__":
    if not os.path.exists(os.path.join(data_dir, "tsdf_train.npy")):
        _ = lidar_to_tsdf_obstacles(training_lidar, (training_labels > 0.5).astype(np.float32), img_dim=64)

    if not os.path.exists(os.path.join(data_dir, "tsdf_test.npy")):
        _ = lidar_to_tsdf_obstacles(testing_lidar, (labels_test > 0.5).astype(np.float32), img_dim=64)

    loss_history, loss_components = train_model()
    tolerance_accuracy, avg_absolute_error, gate_stats = test_model_with_gating(loss_history, loss_components)
    
    model_path = os.path.join(data_dir, "gated_resnet_lidar_model_tsdf_min_max.pth")
    torch.save(resnet.state_dict(), model_path)
    print(f'Model saved to {model_path}')
    
    # Additional analysis
    print("\n=== GATING ANALYSIS ===")
    print(f"The model learned to use min_gate with weight {gate_stats['min_gate_avg']:.3f}")
    print(f"The model learned to use max_gate with weight {gate_stats['max_gate_avg']:.3f}")
    print(f"Average uncertainty: {gate_stats['uncertainty_avg']:.3f}")
    print(f"Average confidence: {gate_stats['confidence_avg']:.3f}")
