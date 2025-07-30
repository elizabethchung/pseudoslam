import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import numpy as np
import os
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


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
    _, axs = plt.subplots(4, 1, figsize=(16, 8))
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
num_epochs = 5
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
    _, axs = plt.subplots(1, 1, figsize=(5, 5))
    axs.imshow(tsdf_map[0, 0], cmap='viridis')
    axs.set_title("TSDF Map")
    plt.tight_layout()
    plt.show()

def labels_to_min_max(labels):
    """
    Detect contiguous regions of 1s in each label row (with wraparound).
    Returns min and max indices for each contiguous range.
    """
    labels = np.atleast_2d(labels)
    min_indices_all, max_indices_all = [], []

    for row in labels:
        row = np.asarray(row)
        n = row.shape[0]

        if np.all(row == 0):
            min_indices_all.append([])
            max_indices_all.append([])
            continue

        # Duplicate row to handle wraparound
        extended = np.concatenate([row, row])
        in_range = False
        ranges = []
        start = None

        for i in range(2 * n):
            val = extended[i]

            if val == 1:
                if not in_range:
                    start = i
                    in_range = True
            else:
                if in_range:
                    end = i - 1
                    if start < n:
                        ranges.append((start % n, end % n))
                    in_range = False

        # If we ended inside a 1-range
        if in_range and start < n:
            end = 2 * n - 1
            ranges.append((start % n, end % n))

        # Merge wraparound if necessary (first starts at 0 and last ends at n-1)
        if len(ranges) > 1 and ranges[0][0] == 0 and ranges[-1][1] == (n - 1):
            merged = (ranges[-1][0], ranges[0][1])
            ranges = [merged] + ranges[1:-1]

        min_indices_all.append([r[0] for r in ranges])
        max_indices_all.append([r[1] for r in ranges])

    return min_indices_all, max_indices_all


def generate_multi_obstacle_targets(labels):
    """
    Generate targets that can handle multiple obstacles per sample
    Returns: list of (min_indices, max_indices) tuples for each sample
    """
    batch_targets = []
    
    for sample_idx in range(labels.shape[0]):
        label_row = labels[sample_idx]
        if not torch.is_tensor(label_row):
            label_row = torch.tensor(label_row, dtype=torch.long)
        
        # Find all contiguous regions of obstacles (1s)
        diff = torch.diff(torch.cat([torch.tensor([0]), label_row, torch.tensor([0])]))
        starts = torch.where(diff == 1)[0]
        ends = torch.where(diff == -1)[0] - 1
        
        # Handle wraparound case
        if label_row[0] == 1 and label_row[-1] == 1:
            # Check if it's one continuous region wrapping around
            if len(starts) > 1:
                # Merge first and last regions
                starts = torch.cat([starts[1:], starts[:1]])
                ends = torch.cat([ends[:-1], ends[-1:]])
        
        obstacle_regions = list(zip(starts.tolist(), ends.tolist()))
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
        
        # Skip if no obstacle regions for this sample
        if not obstacle_regions:
            continue

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

        # Only convert to binary_labels if labels is not already a binary array
        if isinstance(labels, tuple):
            self.min_indices, self.max_indices = labels
            # Check if labels are already binary (e.g., shape (N, 360) and values in {0,1})
            if (
                isinstance(self.min_indices, np.ndarray) and
                self.min_indices.ndim == 2 and
                self.min_indices.shape[1] == 360 and
                np.all((self.min_indices == 0) | (self.min_indices == 1))
            ):
                self.labels = self.min_indices
            else:
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
    Gating module that learns to weight min and max predictions based on input features
    """
    def __init__(self, feature_dim=64, hidden_dim=32):
        super().__init__()
        self.feature_dim = feature_dim
        # Global context network
        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(feature_dim, hidden_dim, 1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, 1),
            nn.ReLU()
        )
        # Local context network
        self.local_context = nn.Sequential(
            nn.Conv1d(feature_dim, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU()
        )
        # Gating network - only outputs 2 values for min/max gates
        self.gate_network = nn.Sequential(
            nn.Conv1d(hidden_dim * 2, hidden_dim, 1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, 2, 1),  # Only 2 outputs for min/max gates
            # No activation here - will use softmax
        )
        
        # Separate uncertainty network
        self.uncertainty_network = nn.Sequential(
            nn.Conv1d(hidden_dim * 2, hidden_dim, 1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, 1, 1),
            nn.Sigmoid()
        )
        
        self.confidence_network = nn.Sequential(
            nn.Conv1d(hidden_dim * 2, hidden_dim, 1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, features):
        batch_size, channels, seq_len = features.shape
        # Extract global and local contexts
        global_ctx = self.global_context(features)  # (batch, hidden_dim, 1)
        global_ctx = global_ctx.expand(-1, -1, seq_len)  # (batch, hidden_dim, 360)
        local_ctx = self.local_context(features)  # (batch, hidden_dim, 360)
        
        # Combine contexts
        combined_ctx = torch.cat([global_ctx, local_ctx], dim=1)  # (batch, hidden_dim*2, 360)
        
        # Generate gates using softmax
        gate_logits = self.gate_network(combined_ctx)  # (batch, 2, 360)
        gates = F.softmax(gate_logits, dim=1)  # Softmax along the gate dimension
        min_gate = gates[:, 0:1, :]  # (batch, 1, 360)
        max_gate = gates[:, 1:2, :]  # (batch, 1, 360)
        
        # Generate uncertainty and confidence separately
        uncertainty = self.uncertainty_network(combined_ctx)  # (batch, 1, 360)
        confidence = self.confidence_network(combined_ctx)  # (batch, 1, 360)
        
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

        # Sanity check shape
        # Remove extra singleton dimensions if present
        while tsdf_data.ndim > 4:
            tsdf_data = tsdf_data.squeeze(0)
        assert tsdf_data.ndim == 4, f"Expected 4D TSDF, got {tsdf_data.shape}"

        # TSDF should be (B, 2, H, 360) → average over H (radial depth)
        if tsdf_data.shape[-1] != 360:
            tsdf_data = F.interpolate(tsdf_data, size=(tsdf_data.shape[2], 360), mode='bilinear', align_corners=False)

        tsdf_radial = tsdf_data.mean(dim=2)  # Result: (B, 2, 360)
        assert tsdf_radial.shape[1] == 2, f"TSDF channels should be 2, got {tsdf_radial.shape[1]}"


        # Step 2: Apply 1D conv to get TSDF angular features (e.g., 128 channels across 360 angles)
        tsdf_feat = self.tsdf_conv1d(tsdf_radial)  # Should output (B, 128, 360)

        # Step 3: Fuse TSDF + LiDAR
        x_fused = torch.cat([x_lidar, tsdf_feat], dim=1)  # (B, 192, 360)

        # Fusion layers
        x_fused = torch.relu(self.fusion_conv1(x_fused))  # (B, 128, 360)
        x_fused = torch.relu(self.fusion_conv2(x_fused))  # (B, 64, 360)
        x_fused = self.dropout(x_fused)

        
        # Generate gating variables
        min_gate, max_gate, uncertainty, confidence = self.gating_module(x_fused)
        
        # Generate separate predictions
        min_logits = self.min_head(x_fused).squeeze(1)  # (batch, 360)
        max_logits = self.max_head(x_fused).squeeze(1)  # (batch, 360)
        
        # Generate gated predictions
        gated_output = self.gated_output(x_fused)  # (batch, 2, 360)
        gated_min_logits = gated_output[:, 0, :]  # (batch, 360)
        gated_max_logits = gated_output[:, 1, :]  # (batch, 360)
        
        # Apply gating
        # Method 1: Gate the logits directly
        final_min_logits = min_gate.squeeze(1) * min_logits + (1 - min_gate.squeeze(1)) * gated_min_logits
        final_max_logits = max_gate.squeeze(1) * max_logits + (1 - max_gate.squeeze(1)) * gated_max_logits
        
        # Method 2: Apply uncertainty weighting
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

def gated_min_max_loss(min_pred_logits, max_pred_logits, min_target, max_target, 
                      gating_info, alpha=0.5, beta=0.1):
    """
    Enhanced loss function with gating regularization
    """
    # Standard min-max loss
    min_probs = F.softmax(min_pred_logits, dim=1)
    max_probs = F.softmax(max_pred_logits, dim=1)
    
    angles = torch.arange(360, device=min_pred_logits.device, dtype=torch.float32)
    min_expected = torch.sum(min_probs * angles, dim=1)
    max_expected = torch.sum(max_probs * angles, dim=1)
    
    min_target_float = min_target.float()
    max_target_float = max_target.float()
    
    min_dist = circular_distance(min_expected, min_target_float)
    max_dist = circular_distance(max_expected, max_target_float)
    
    target_span = calculate_span(min_target, max_target)
    predicted_span = calculate_span(min_expected.round().long(), max_expected.round().long())
    span_loss = F.mse_loss(predicted_span.float(), target_span.float())
    
    position_loss = F.mse_loss(min_dist, torch.zeros_like(min_dist)) + \
                   F.mse_loss(max_dist, torch.zeros_like(max_dist))
    
    ce_loss = F.cross_entropy(min_pred_logits, min_target.squeeze()) + \
              F.cross_entropy(max_pred_logits, max_target.squeeze())
    
    standard_loss = alpha * position_loss + (1-alpha) * ce_loss + 0.1 * span_loss
    
    # Gating regularization
    min_gate = gating_info['min_gate']
    max_gate = gating_info['max_gate']
    uncertainty = gating_info['uncertainty']
    confidence = gating_info['confidence']
    
    # Encourage confident predictions for easier samples
    confidence_loss = F.mse_loss(confidence.mean(), torch.ones_like(confidence.mean()) * 0.8)
    
    # Regularize gating to avoid extreme values
    gate_reg = torch.mean(torch.abs(min_gate - 0.5)) + torch.mean(torch.abs(max_gate - 0.5))
    
    # Uncertainty should be lower for confident predictions
    uncertainty_reg = F.mse_loss(uncertainty * confidence, torch.zeros_like(uncertainty * confidence))
    
    # Total loss
    total_loss = standard_loss + beta * (confidence_loss + gate_reg + uncertainty_reg)
    
    return total_loss, {
        'standard_loss': standard_loss.item(),
        'confidence_loss': confidence_loss.item(),
        'gate_reg': gate_reg.item(),
        'uncertainty_reg': uncertainty_reg.item()
    }

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

def debug_dataset_labels(dataset, num_samples=5):
    print("Debugging dataset labels (showing first few samples)...")
    for i in range(num_samples):
        sample = dataset[i]
        lidar_data = sample[0]
        labels = sample[1]  # Assuming labels are (min_indices, max_indices) tuple or tensor
        
        print(f"Sample {i+1}:")
        print(f"  LiDAR shape: {lidar_data.shape if hasattr(lidar_data, 'shape') else type(lidar_data)}")
        
        # If labels is a tuple/list of (min, max)
        if isinstance(labels, (tuple, list)) and len(labels) == 2:
            min_label, max_label = labels
            print(f"  Min indices: {min_label} (type: {type(min_label)}, shape: {getattr(min_label, 'shape', 'N/A')})")
            print(f"  Max indices: {max_label} (type: {type(max_label)}, shape: {getattr(max_label, 'shape', 'N/A')})")
        else:
            print(f"  Labels: {labels} (type: {type(labels)}, shape: {getattr(labels, 'shape', 'N/A')})")

def create_biased_mask(labels, keep_1_prob=1.0, keep_0_prob=0.1):
    """
    Creates a binary mask that keeps:
    - 1s with probability `keep_1_prob`
    - 0s with probability `keep_0_prob`
    """
    mask = np.zeros_like(labels, dtype=np.float32)
    
    ones = (labels == 1)
    zeros = (labels == 0)

    mask[ones] = np.random.binomial(1, keep_1_prob, size=ones.sum())
    mask[zeros] = np.random.binomial(1, keep_0_prob, size=zeros.sum())
    
    return mask

training_mask = create_biased_mask(training_labels, keep_1_prob=1.0, keep_0_prob=0.1)
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

debug_dataset_labels(train_dataset, num_samples=5)
debug_dataset_labels(test_dataset, num_samples=5)

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
        print(f"Number of batches in test loader: {len(test_loader)}")
        print(f"Batch size: {test_loader.batch_size}")
        print(f"Total samples: {len(test_loader.dataset)}")
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
            
            if batch_idx % 1 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], '
                      f'Loss: {loss.item():.4f}')
        
        scheduler.step()
        avg_epoch_loss = epoch_loss / len(train_loader)
        loss_history.append(avg_epoch_loss)
        
        for key in epoch_components:
            loss_components[key].append(epoch_components[key] / len(train_loader))
        
        print(f'Epoch [{epoch+1}/{num_epochs}] Average Loss: {avg_epoch_loss:.4f}')
    
    return loss_history, loss_components

def find_obstacle_ranges(min_logits, max_logits, threshold=0.5):
    batch_size = min_logits.shape[0]
    obstacles_batch = []
    for b in range(batch_size):
        min_scores = min_logits[b].cpu().numpy()
        max_scores = max_logits[b].cpu().numpy()
        
        # Find peaks above threshold
        min_peaks, _ = find_peaks(min_scores, height=threshold)
        max_peaks, _ = find_peaks(max_scores, height=threshold)
        
        # Pair min and max peaks into ranges
        ranges = []
        for min_p in min_peaks:
            # Find max peak that comes after min_p, circularly
            max_candidates = max_peaks[max_peaks >= min_p]
            if len(max_candidates) == 0:
                max_p = max_peaks[0]  # wraparound case
            else:
                max_p = max_candidates[0]
            ranges.append((min_p, max_p))
        obstacles_batch.append(ranges)
    return obstacles_batch

def get_contiguous_ranges(binary_mask):
    """
    Given a 1D binary mask (numpy array),
    returns list of (start_idx, end_idx) tuples for contiguous runs of 1s,
    handling wraparound if needed.
    """
    n = len(binary_mask)
    if np.sum(binary_mask) == 0:
        return []
    
    # Double mask for wraparound detection
    doubled = np.concatenate([binary_mask, binary_mask])
    
    ranges = []
            # length = end - start + 1
    start = 0
    for i in range(2 * n):
        if doubled[i] == 1 and not in_range:
            in_range = True
            start = i
        elif doubled[i] == 0 and in_range:
                    # first_start, first_end = ranges[0]
            end = i - 1
            length = end - start + 1
            if start < n:
                # Normal range
                ranges.append((start, min(end, n - 1)))
            else:
                # Wraparound range: merge with first if exists
                if len(ranges) > 0:
                    first_start, first_end = ranges[0]
                    ranges[0] = ((start - n), first_end)
                else:
                    ranges.append((start - n, min(end - n, n - 1)))
            if len(ranges) > 0:
                break
    # If ended inside range at the end of doubled array
    if in_range:
        if len(ranges) > 0:
            first_start, first_end = ranges[0]
            ranges[0] = (0, first_end)
        else:
            ranges.append((0, n-1))
    return ranges


def post_process_logits(pred_min_logits, pred_max_logits, threshold=0.5):
    """
    Inputs:
        pred_min_logits, pred_max_logits: numpy arrays of shape (batch, 360) - logits per angle
        threshold: probability threshold to detect edges
    Returns:
        list of obstacle spans per sample, each as list of (min_idx, max_idx) tuples
    """
    batch_size = pred_min_logits.shape[0]
    obstacle_spans_batch = []
    
    # Convert logits to probabilities
    min_probs = 1 / (1 + np.exp(-pred_min_logits))
    max_probs = 1 / (1 + np.exp(-pred_max_logits))
    
    for b in range(batch_size):
        min_mask = (min_probs[b] > threshold).astype(int)
        max_mask = (max_probs[b] > threshold).astype(int)
        
        # Get contiguous ranges of min and max separately
        min_ranges = get_contiguous_ranges(min_mask)
        max_ranges = get_contiguous_ranges(max_mask)
        
        # Now pair min and max ranges based on proximity (simple heuristic)
        # We'll pair ith min with ith max if same number, else fallback
        
        spans = []
        if len(min_ranges) == len(max_ranges):
            for i in range(len(min_ranges)):
                spans.append((min_ranges[i][0], max_ranges[i][1]))
        else:
            # Fallback: pair first min with first max, etc, truncated to min length
            pairs = min(len(min_ranges), len(max_ranges))
            for i in range(pairs):
                spans.append((min_ranges[i][0], max_ranges[i][1]))
        
        obstacle_spans_batch.append(spans)
    
    return obstacle_spans_batch


def test_model_with_gating(loss_history, loss_components, peak_threshold=0.7):
    resnet.eval()
    total_samples = 0
    total_obstacles = 0
    matched_obstacles = 0
    total_absolute_error = []

    tolerance = 10  # degrees

    gate_stats = {
        'min_gate_avg': 0.0,
        'max_gate_avg': 0.0,
        'uncertainty_avg': 0.0,
        'confidence_avg': 0.0,
    }
    num_batches = 0

    for batch_idx, (distances, labels, _, tsdf) in enumerate(test_loader, start=1):
        # Generate ground-truth obstacle spans for this batch
        gt_obstacle_targets = generate_multi_obstacle_targets(labels)

        if batch_idx % 1 == 0:
            # Print first sample's labels and generated targets for debugging
            print(f'Batch {batch_idx} Sample 10 Scan:', distances[9].cpu().numpy())
            print(f"Batch {batch_idx} Sample 10 Labels:", labels[9])
            print(f"Batch {batch_idx} Sample 10 GT Obstacle Targets:", gt_obstacle_targets[9])


        distances = distances.to(device)
        tsdf = tsdf.to(device)

        # Forward pass
        pred_min_logits, pred_max_logits, gating_info = resnet(distances, tsdf)

        # Debug prints (optional, keep or comment out if too verbose)
        # print(f"Batch {batch_idx}/{len(test_loader)}:")
        # print("pred_mins shape:", pred_min_logits.shape)
        # print("pred_maxs shape:", pred_max_logits.shape)
        # for key, tensor in gating_info.items():
        #     print(f"{key} shape: {tensor.shape}")

        # Apply sigmoid to get scores for peak detection
        pred_min_scores = torch.sigmoid(pred_min_logits).cpu().detach().numpy()
        pred_max_scores = torch.sigmoid(pred_max_logits).cpu().detach().numpy()

        batch_size = distances.size(0)
        total_samples += batch_size

        for b in range(batch_size):
            gt_spans = gt_obstacle_targets[b]

            # Find peaks above threshold for min and max logits
            min_peaks, _ = find_peaks(pred_min_scores[b], height=peak_threshold)
            max_peaks, _ = find_peaks(pred_max_scores[b], height=peak_threshold)

            # Pair min peaks to max peaks to form predicted obstacle ranges
            pred_spans = []
            for min_p in min_peaks:
                # Find max peak that comes after min_p circularly
                max_candidates = max_peaks[max_peaks >= min_p]
                if len(max_candidates) == 0 and len(max_peaks) > 0:
                    max_p = max_peaks[0]  # wraparound case
                elif len(max_candidates) > 0:
                    max_p = max_candidates[0]
                else:
                    continue
                pred_spans.append((min_p, max_p))

            total_obstacles += len(gt_spans)

            for gt_min, gt_max in gt_spans:
                best_error = float('inf')
                for pred_min, pred_max in pred_spans:
                    gt_width = (gt_max - gt_min + 360) % 360
                    pred_width = (pred_max - pred_min + 360) % 360
                    error = abs(pred_width - gt_width)
                    if error < best_error:
                        best_error = error

                total_absolute_error.append(best_error)
                if best_error <= tolerance:
                    matched_obstacles += 1

            # Print predicted and target spans for batch 8
            if batch_idx == 7:
                print(f"Batch {batch_idx} Sample {b+1}/{batch_size} Predicted Spans: {pred_spans}")
                print(f"Batch {batch_idx} Sample {b+1}/{batch_size} Ground Truth Spans: {gt_spans}")

        # Aggregate gating stats
        gate_stats['min_gate_avg'] += gating_info['min_gate'].mean().item()
        gate_stats['max_gate_avg'] += gating_info['max_gate'].mean().item()
        gate_stats['uncertainty_avg'] += gating_info['uncertainty'].mean().item()
        gate_stats['confidence_avg'] += gating_info['confidence'].mean().item()
        num_batches += 1

    avg_error = np.mean(total_absolute_error)
    obstacle_accuracy = 100.0 * matched_obstacles / total_obstacles

    print(f'Total GT Obstacles: {total_obstacles}')
    print(f'Matched Obstacles (±{tolerance}° width): {matched_obstacles}')
    print(f'Obstacle Detection Accuracy: {obstacle_accuracy:.2f}%')
    print(f'Average Width Error: {avg_error:.2f} degrees')

    print(f'Gating Statistics:')
    print(f'  Min Gate Average: {gate_stats["min_gate_avg"] / num_batches:.3f}')
    print(f'  Max Gate Average: {gate_stats["max_gate_avg"] / num_batches:.3f}')
    print(f'  Uncertainty Average: {gate_stats["uncertainty_avg"] / num_batches:.3f}')
    print(f'  Confidence Average: {gate_stats["confidence_avg"] / num_batches:.3f}')
    
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
    gate_values = [v / num_batches for v in gate_stats.values()]
    plt.bar(gate_names, gate_values)
    plt.title('Gating Statistics')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    return obstacle_accuracy, avg_error, gate_stats




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
