import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import numpy as np
import os
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from model_testing import test_model_with_gating_loader, test_model_with_gating_single_sample


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
num_epochs = 30
batch_size = 64
learning_rate = 0.001
torch.backends.cudnn.benchmark = True  # Enable cuDNN auto-tuner for faster training

torch.autograd.set_detect_anomaly(True)
torch.cuda.empty_cache()


def get_obstacle_counts(batch_labels):
    if isinstance(batch_labels, np.ndarray):
        batch_labels = torch.tensor(batch_labels)

    if batch_labels.ndim == 1:
        batch_labels = batch_labels.unsqueeze(0)  # (360,) → (1, 360)

    counts = []
    for row in batch_labels:
        row = row.int()
        extended = torch.cat([row, row[:1]])
        diff = torch.diff(extended)
        starts = (diff == 1).sum()
        ends = (diff == -1).sum()
        count = min(starts, ends)

        if row.sum() == 360:
            count = 1

        counts.append(count)

    return torch.tensor(counts, dtype=torch.float32).unsqueeze(1)

def is_flat(lidar, i, threshold=1.0):
    return abs(lidar[(i+1)%360] + lidar[(i-1)%360] - 2*lidar[i]) < threshold


def lidar_to_tsdf_obstacles(lidar_data, img_dim=32, max_range=125.0, trunc_thresh=3):
    """
    LIDAR to TSDF conversion with obstacle, free-space, and unknown channels.
    Unexplored regions are NOT treated as obstacles.
    """
    if lidar_data.ndim == 1:
        lidar_data = np.expand_dims(lidar_data, axis=0)
    
    batch_size = lidar_data.shape[0]
    center = img_dim // 2
    scale = img_dim / (2 * max_range)  # meters to pixels
    angles = np.deg2rad(np.arange(360))
    
    y_indices, x_indices = np.mgrid[0:img_dim, 0:img_dim]
    
    obstacle_tsdf = np.full((batch_size, img_dim, img_dim), trunc_thresh, dtype=np.float32)
    free_space_tsdf = np.full((batch_size, img_dim, img_dim), trunc_thresh, dtype=np.float32)
    explored_mask = np.zeros((batch_size, img_dim, img_dim), dtype=bool)
    print("Using max_range:", max_range)
    print("LiDAR max:", np.max(lidar_data))
    print("Scale factor:", scale)


    for b in range(batch_size):
        valid_rays = lidar_data[b] > 0  # Keep rays even if they exceed max_range
        
        for i in range(360):
            if not valid_rays[i]:
                continue
            
            distance = lidar_data[b, i]
            angle = angles[i]
            
            # Clamp ray length to max_range
            effective_distance = min(distance, max_range)
            x_end = effective_distance * np.cos(angle)
            y_end = effective_distance * np.sin(angle)
            x_end_grid = center + x_end * scale
            y_end_grid = center + y_end * scale
            
            if 0 <= x_end_grid < img_dim and 0 <= y_end_grid < img_dim:
                # Mark endpoint as explored even if it's just the end of the ray
                explored_mask[b][int(y_end_grid), int(x_end_grid)] = True

                # Cast ray regardless of hit/miss
                ray_steps = int(effective_distance * scale * 1.2)
                if ray_steps > 1:
                    for step in range(1, ray_steps):
                        t = step / ray_steps
                        x_ray = t * x_end
                        y_ray = t * y_end
                        x_ray_grid = center + x_ray * scale
                        y_ray_grid = center + y_ray * scale
                        
                        if 0 <= x_ray_grid < img_dim and 0 <= y_ray_grid < img_dim:
                            dist_to_ray = np.sqrt((x_indices - x_ray_grid)**2 + (y_indices - y_ray_grid)**2)
                            free_sdf = dist_to_ray - 2.0
                            free_sdf = np.clip(free_sdf, -trunc_thresh, trunc_thresh)
                            free_space_tsdf[b] = np.minimum(free_space_tsdf[b], free_sdf)
                            explored_mask[b][dist_to_ray <= 2.0] = True

            # Only mark obstacle if distance < max_range
            if distance < max_range:
                def is_flat(lidar, idx, threshold=1.3):
                    prev = lidar[(idx - 1) % 360]
                    next = lidar[(idx + 1) % 360]
                    return abs(prev + next - 2 * lidar[idx]) < threshold
                
                if is_flat(lidar_data[b], i):
                    continue

                x_obs = distance * np.cos(angle)
                y_obs = distance * np.sin(angle)
                x_obs_grid = center + x_obs * scale
                y_obs_grid = center + y_obs * scale
                
                if 0 <= x_obs_grid < img_dim and 0 <= y_obs_grid < img_dim:
                    dist_to_obs = np.sqrt((x_indices - x_obs_grid)**2 + (y_indices - y_obs_grid)**2)
                    obs_sdf = dist_to_obs - 1.5
                    obs_sdf = np.clip(obs_sdf, -trunc_thresh, trunc_thresh)
                    obstacle_tsdf[b] = np.minimum(obstacle_tsdf[b], obs_sdf)
                    explored_mask[b][dist_to_obs <= 1.5] = True

    unknown_channel = (~explored_mask).astype(np.float32)
    # Final masking to ensure clean outputs
    obstacle_tsdf = np.where(explored_mask, obstacle_tsdf, 0)
    free_space_tsdf = np.where(explored_mask, free_space_tsdf, 0)
    result = np.stack([obstacle_tsdf, free_space_tsdf, unknown_channel], axis=1)
    return result


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

def improved_multi_obstacle_loss(predictions, obstacle_targets, 
                                 alpha=0.5, beta=0.1, count_weight=3.0, 
                                 confidence_weight=0.05):
    """
    Improved loss function for multi-obstacle detection.
    Now with proper gate regularization to push gates away from 0.5.
    """
    min_pred_logits = predictions['min_logits']
    max_pred_logits = predictions['max_logits']
    confidence_map = predictions['confidence']
    count_logits = predictions['count']

    batch_size = min_pred_logits.shape[0]
    total_loss = 0.0

    components = {
        'position_loss': 0.0,
        'count_loss': 0.0,
        'confidence_loss': 0.0,
        'gate_reg': 0.0,
        'uncertainty_reg': 0.0
    }

    true_counts = []
    for batch_idx in range(batch_size):
        obstacle_regions = obstacle_targets[batch_idx]
        true_count = len(obstacle_regions)
        true_counts.append(true_count)

        # Position & confidence losses
        if obstacle_regions:
            sample_position_loss = 0.0
            confidence_targets = torch.zeros_like(confidence_map[batch_idx])

            for region in obstacle_regions:
                min_target = torch.tensor(region[0], device=min_pred_logits.device)
                max_target = torch.tensor(region[1], device=min_pred_logits.device)

                ce_min = F.cross_entropy(min_pred_logits[batch_idx:batch_idx+1], min_target.unsqueeze(0))
                ce_max = F.cross_entropy(max_pred_logits[batch_idx:batch_idx+1], max_target.unsqueeze(0))
                sample_position_loss += ce_min + ce_max

                confidence_targets[region[0]:region[1]+1] = 1.0

            components['position_loss'] += sample_position_loss / len(obstacle_regions)

            conf_loss = F.binary_cross_entropy(confidence_map[batch_idx], confidence_targets)
            components['confidence_loss'] += conf_loss
            total_loss += confidence_weight * conf_loss

        else:
            conf_loss = F.binary_cross_entropy(confidence_map[batch_idx], 
                                               torch.zeros_like(confidence_map[batch_idx]))
            components['confidence_loss'] += conf_loss
            total_loss += confidence_weight * conf_loss

    # Count loss
    true_counts_tensor = torch.tensor(true_counts, dtype=torch.long, device=count_logits.device)
    num_classes = count_logits.shape[1]
    true_counts_tensor = torch.clamp(true_counts_tensor, 0, num_classes - 1)
    count_loss = F.cross_entropy(count_logits, true_counts_tensor)
    components['count_loss'] = count_loss.item()
    total_loss += count_weight * count_loss

    # Normalize position loss
    if components['position_loss'] > 0:
        total_loss += alpha * components['position_loss'] / batch_size
        components['position_loss'] /= batch_size

    # Better gate and uncertainty regularization
    if 'min_gate' in predictions:
        min_gate = predictions['min_gate']  # [B, N]
        max_gate = predictions['max_gate']  # [B, N]
        uncertainty = predictions['uncertainty']  # [B, N]
        confidence = predictions['confidence']  # [B, N]

        # Gate regularization: encourage gates to be near 0 or 1 (avoid 0.5)
        gate_reg = torch.mean((min_gate - 0.5)**2 + (max_gate - 0.5)**2)
        components['gate_reg'] = gate_reg.item()

        # Uncertainty regularization: confidence-weighted
        uncertainty_reg = torch.mean(uncertainty * confidence)
        components['uncertainty_reg'] = uncertainty_reg.item()

        total_loss += beta * (gate_reg + uncertainty_reg)

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
            self.tsdf_maps = lidar_to_tsdf_obstacles(self.data)
            if tsdf_cache_path:
                np.save(tsdf_cache_path, self.tsdf_maps)
                print(f"Saved TSDF maps to {tsdf_cache_path}")
        # print("Computing TSDF maps...")
        # self.tsdf_maps = lidar_to_tsdf_obstacles(self.data)
        # if tsdf_cache_path:
        #     np.save(tsdf_cache_path, self.tsdf_maps)
        #     print(f"Saved TSDF maps to {tsdf_cache_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        lidar_data = self.data[idx]
        label = self.labels[idx]
        tsdf_tensor = torch.from_numpy(self.tsdf_maps[idx]).float()  # (1, 64, 64)

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
    Gating module that learns to output up to 3 obstacle min/max pairs
    and gates the appropriate number of obstacles based on input features
    """
    def __init__(self, feature_dim=64, hidden_dim=32, max_obstacles=3):
        super().__init__()
        self.feature_dim = feature_dim
        self.max_obstacles = max_obstacles
        
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
        
        # Obstacle count network - predicts how many obstacles are present
        self.obstacle_count_network = nn.Sequential(
            nn.Conv1d(hidden_dim * 2, hidden_dim, 1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, max_obstacles + 1, 1),  # +1 for "no obstacle" option
            # No activation - will use softmax
        )
        
        # Obstacle presence gates - binary gate for each obstacle
        self.presence_gates = nn.ModuleList([
            nn.Sequential(
            nn.Conv1d(hidden_dim * 2, hidden_dim, 1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim // 2, 1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim // 2, 1, 1),
            nn.Sigmoid()
        ) for _ in range(max_obstacles)
        ])
        
        # Min/Max gates for each obstacle
        self.minmax_gates = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(hidden_dim * 2, hidden_dim, 1),
                nn.ReLU(),
                nn.Conv1d(hidden_dim, 2, 1),  # 2 outputs for min/max gates
                # No activation - will use softmax
            ) for _ in range(max_obstacles)
        ])
        
        # Uncertainty network for each obstacle
        self.uncertainty_networks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(hidden_dim * 2, hidden_dim, 1),
                nn.ReLU(),
                nn.Conv1d(hidden_dim, 1, 1),
                nn.Sigmoid()
            ) for _ in range(max_obstacles)
        ])
        
        # Confidence network for each obstacle
        self.confidence_networks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(hidden_dim * 2, hidden_dim, 1),
                nn.ReLU(),
                nn.Conv1d(hidden_dim, 1, 1),
                nn.Sigmoid()
            ) for _ in range(max_obstacles)
        ])

    def forward(self, features):
        batch_size, channels, seq_len = features.shape
        
        # Extract global and local contexts
        global_ctx = self.global_context(features)  # (batch, hidden_dim, 1)
        global_ctx = global_ctx.expand(-1, -1, seq_len)  # (batch, hidden_dim, 360)
        local_ctx = self.local_context(features)  # (batch, hidden_dim, 360)
        
        # Combine contexts
        combined_ctx = torch.cat([global_ctx, local_ctx], dim=1)  # (batch, hidden_dim*2, 360)
        
        # # Predict obstacle count distribution
        obstacle_count_logits = self.obstacle_count_network(combined_ctx)  # (batch, max_obstacles+1, 360)
        obstacle_count_probs = F.softmax(obstacle_count_logits, dim=1)  # Softmax along count dimension
        
        # Generate gates for each obstacle
        obstacle_gates = {}
        for i in range(self.max_obstacles):
            # Presence gate
            presence_gate = self.presence_gates[i](combined_ctx)  # (batch, 1, 360)
            
            # Min/Max gates
            minmax_logits = self.minmax_gates[i](combined_ctx)  # (batch, 2, 360)
            minmax_gates = F.softmax(minmax_logits, dim=1)  # Softmax along min/max dimension
            min_gate = minmax_gates[:, 0:1, :]  # (batch, 1, 360)
            max_gate = minmax_gates[:, 1:2, :]  # (batch, 1, 360)
            
            # Uncertainty and confidence
            uncertainty = self.uncertainty_networks[i](combined_ctx)  # (batch, 1, 360)
            confidence = self.confidence_networks[i](combined_ctx)  # (batch, 1, 360)
            
            obstacle_gates[f'obstacle_{i}'] = {
                'presence_gate': presence_gate,
                'min_gate': min_gate,
                'max_gate': max_gate,
                'uncertainty': uncertainty,
                'confidence': confidence
            }
        
        return obstacle_gates #, obstacle_count_probs

class ImprovedResNetWithGating(nn.Module):
    def __init__(self, max_obstacles=3):
        super().__init__()
        self.max_obstacles = max_obstacles
        
        # LiDAR processing (assuming ResidualBlock is defined elsewhere)
        self.lidar_layer1 = ResidualBlock(1, 16)
        self.lidar_layer2 = ResidualBlock(16, 32)
        self.lidar_layer3 = ResidualBlock(32, 64)
        
        # TSDF Conv1D path
        self.tsdf_conv1d = nn.Sequential(
            nn.Conv1d(3, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2)
        )

        self.tsdf_conv2d = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),  # 3-channel input
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU()
        )


        # Fusion layers
        self.fusion_conv1 = nn.Conv1d(96, 128, kernel_size=3, padding=1)
        self.fusion_conv2 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
        
        # Multi-obstacle gating module
        self.gating_module = GatingModule(feature_dim=64, hidden_dim=32, max_obstacles=max_obstacles)
        
        # Separate heads for min and max for each obstacle
        self.obstacle_heads = nn.ModuleList([
            nn.ModuleDict({
                'min_head': nn.Sequential(
                    nn.Conv1d(64, 32, 1),
                    nn.ReLU(),
                    nn.Conv1d(32, 1, 1)
                ),
                'max_head': nn.Sequential(
                    nn.Conv1d(64, 32, 1),
                    nn.ReLU(),
                    nn.Conv1d(32, 1, 1)
                )
            }) for _ in range(max_obstacles)
        ])
        
        # Gated output heads for each obstacle
        self.gated_outputs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(64, 32, 1),
                nn.ReLU(),
                nn.Conv1d(32, 2, 1)  # 2 outputs: min and max
            ) for _ in range(max_obstacles)
        ])
        
        self.dropout = nn.Dropout(0.1)

        # Obstacle count head (predicts how many distinct obstacles)
        self.count_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(4),
            nn.Flatten(),
            nn.Linear(64*4, 32),
            nn.ReLU(),
            nn.Linear(32, 4)  # 4 classes: 0, 1, 2, 3 obstacles
        )
                
    def forward(self, lidar_data, tsdf_data):
        # Process LiDAR
        x_lidar = self.lidar_layer1(lidar_data)  # (B, 16, 360)
        x_lidar = self.lidar_layer2(x_lidar)     # (B, 32, 360)
        x_lidar = self.lidar_layer3(x_lidar)     # (B, 64, 360)

        # Process TSDF
        while tsdf_data.ndim > 4:
            tsdf_data = tsdf_data.squeeze(0)
        assert tsdf_data.ndim == 4, f"Expected 4D TSDF, got {tsdf_data.shape}"

        if tsdf_data.shape[-1] != 360:
            tsdf_data = F.interpolate(tsdf_data, size=(tsdf_data.shape[2], 360), mode='bilinear', align_corners=False)

        tsdf_feat = self.tsdf_conv2d(tsdf_data)  # (B, 32, 32, 32)
        tsdf_feat = F.adaptive_avg_pool2d(tsdf_feat, (1, 360))  # preserves angle dimension
        tsdf_feat = tsdf_feat.squeeze(2)  # shape: (B, 32, 360)

        # Ensure TSDF features have shape (B, 128, 360) before concatenation
        if tsdf_feat.shape[-1] != x_lidar.shape[-1]:
            tsdf_feat = F.interpolate(tsdf_feat, size=x_lidar.shape[-1], mode='linear', align_corners=False)
        # Fuse TSDF + LiDAR
        x_fused = torch.cat([x_lidar, tsdf_feat], dim=1)  # (B, 96, 360)
        x_fused = torch.relu(self.fusion_conv1(x_fused))  # (B, 128, 360)
        x_fused = torch.relu(self.fusion_conv2(x_fused))  # (B, 64, 360)
        x_fused = self.dropout(x_fused)

        # print("x_fused shape before count_head:", x_fused.shape)


        # # Count prediction: output a single scalar per sample (regression)
        # x = self.count_head[0](x_fused)  # AdaptiveAvgPool1d(4)
        # print("After pooling:", x.shape)  # Should be (64, 64, 4)
        # x = self.count_head[1](x)  # Flatten
        # print("After flatten:", x.shape) # Should be (64, 256)
        # x = self.count_head[2](x)  # Linear 256->64
        # print("After linear1:", x.shape) # Should be (64, 64)
        # x = self.count_head[3](x)  # ReLU
        # x = self.count_head[4](x)  # Linear 64->4
        # print("After linear2:", x.shape) # Should be (64, 4)
        # count_logits = x  # (B, 4) - logits for 0, 1, 2, 3 obstacles
        count_logits = self.count_head(x_fused)  # (B, 4) - logits for 0, 1, 2, 3 obstacles
        count_pred = count_logits  # Use raw logits for loss (cross-entropy expects logits)
        # Generate gating variables
        obstacle_gates = self.gating_module(x_fused)
        # Initialize combined outputs
        batch_size = x_fused.shape[0]
        combined_confidence = torch.zeros(batch_size, 360, device=x_fused.device)
        all_min_logits = []
        all_max_logits = []
        
        # Process each obstacle
        obstacle_outputs = []
        for i in range(self.max_obstacles):
            gates = obstacle_gates[f'obstacle_{i}']
            presence_gate = gates['presence_gate']
            min_gate = gates['min_gate'] 
            max_gate = gates['max_gate']
            uncertainty = gates['uncertainty']
            confidence = gates['confidence']
            
            # Generate predictions for this obstacle
            min_logits = self.obstacle_heads[i]['min_head'](x_fused).squeeze(1)  # (B, 360)
            max_logits = self.obstacle_heads[i]['max_head'](x_fused).squeeze(1)  # (B, 360)
            
            # Generate gated predictions
            gated_output = self.gated_outputs[i](x_fused)  # (B, 2, 360)
            gated_min_logits = gated_output[:, 0, :]
            gated_max_logits = gated_output[:, 1, :]
            
            # Apply gating
            gated_min = min_gate.squeeze(1) * min_logits + (1 - min_gate.squeeze(1)) * gated_min_logits
            gated_max = max_gate.squeeze(1) * max_logits + (1 - max_gate.squeeze(1)) * gated_max_logits
            
            # Apply uncertainty weighting
            final_min = gated_min * (1 - uncertainty.squeeze(1)) + uncertainty.squeeze(1) * gated_min_logits
            final_max = gated_max * (1 - uncertainty.squeeze(1)) + uncertainty.squeeze(1) * gated_max_logits
            
            # Apply presence gating
            final_min = final_min * presence_gate.squeeze(1)
            final_max = final_max * presence_gate.squeeze(1)
            
            # CRITICAL: Combine confidence maps from all obstacles
            obstacle_confidence = confidence.squeeze(1) * presence_gate.squeeze(1)
            combined_confidence = torch.max(combined_confidence, obstacle_confidence)
            
            all_min_logits.append(final_min)
            all_max_logits.append(final_max)
            
            obstacle_outputs.append({
                'min_logits': final_min,
                'max_logits': final_max,
                'presence_gate': presence_gate,
                'min_gate': min_gate,
                'max_gate': max_gate,
                'uncertainty': uncertainty,
                'confidence': confidence,
                'raw_min_logits': min_logits,
                'raw_max_logits': max_logits
            })
        
        # Create weighted combination of predictions based on presence gates
        presence_weights = torch.stack([gates['presence_gate'].squeeze(1) 
                                    for gates in [obstacle_gates[f'obstacle_{i}'] 
                                                for i in range(self.max_obstacles)]], dim=1)  # (B, max_obstacles, 360)
        
        # Normalize presence weights
        presence_weights = F.softmax(presence_weights, dim=1)
        
        # Combine min/max logits using presence weights
        stacked_min = torch.stack(all_min_logits, dim=1)  # (B, max_obstacles, 360)
        stacked_max = torch.stack(all_max_logits, dim=1)  # (B, max_obstacles, 360)
        
        combined_min_logits = (stacked_min * presence_weights).sum(dim=1)  # (B, 360)
        combined_max_logits = (stacked_max * presence_weights).sum(dim=1)  # (B, 360)
        
        # Return format that matches your loss function expectations
        return {
            'min_logits': combined_min_logits,
            'max_logits': combined_max_logits, 
            'confidence': combined_confidence,
            'count': count_pred,
            'min_gate': torch.stack([gates['min_gate'] for gates in [obstacle_gates[f'obstacle_{i}'] for i in range(self.max_obstacles)]], dim=1).mean(dim=1),
            'max_gate': torch.stack([gates['max_gate'] for gates in [obstacle_gates[f'obstacle_{i}'] for i in range(self.max_obstacles)]], dim=1).mean(dim=1),
            'uncertainty': torch.stack([gates['uncertainty'] for gates in [obstacle_gates[f'obstacle_{i}'] for i in range(self.max_obstacles)]], dim=1).mean(dim=1),
            'individual_obstacles': obstacle_outputs  # Keep individual outputs for analysis
            }
    

# Load data
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, "data")
training_lidar = np.load(os.path.join(data_dir, "fix_training_lidar.npy"))
training_labels = np.load(os.path.join(data_dir, "fix_training_labels.npy"))
testing_lidar = np.load(os.path.join(data_dir, "fix_testing_lidar.npy"))
labels_test = np.load(os.path.join(data_dir, "fix_testing_labels.npy"))


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
    tsdf_cache_path=os.path.join(data_dir, "fix_tsdf_train.npy")
)

test_dataset = LidarTSDFDataset(
    testing_lidar, 
    (testing_min_indices, testing_max_indices), 
    testing_mask,
    tsdf_cache_path=os.path.join(data_dir, "fix_tsdf_test.npy")
)

# debug_dataset_labels(train_dataset, num_samples=5)
# debug_dataset_labels(test_dataset, num_samples=5)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)


# Use improved model with gating
resnet = ImprovedResNetWithGating().to(device)

def train_model():
    optimizer = optim.Adam(resnet.parameters(), lr=0.0005, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    loss_history = []
    loss_components = {
        'standard': [], 
        'confidence': [], 
        'gate_reg': [], 
        'uncertainty_reg': [],
        'count': []  # NEW
    }

    for epoch in range(num_epochs):
        print(f"Number of batches in train loader: {len(train_loader)}")
        print(f"Batch size: {train_loader.batch_size}")
        print(f"Total samples: {len(train_loader.dataset)}")

        resnet.train()
        epoch_loss = 0.0
        epoch_components = {
            'standard': 0.0, 
            'confidence': 0.0, 
            'gate_reg': 0.0, 
            'uncertainty_reg': 0.0,
            'count': 0.0
        }

        for batch_idx, (distances, labels, _, tsdf) in enumerate(train_loader):
            distances = distances.to(device)
            tsdf = tsdf.to(device)

            obstacle_targets = generate_multi_obstacle_targets(labels)
            obstacle_counts_gt = get_obstacle_counts(labels).to(device)  # (B,)

            optimizer.zero_grad()
            predictions = resnet(distances, tsdf)

            # Extract components
            min_pred = predictions['min_logits']
            max_pred = predictions['max_logits']
            confidence_map = predictions['confidence']
            count_preds = predictions['count']  # predictions
            gating_info = {
                'min_gate': predictions['min_gate'],
                'max_gate': predictions['max_gate'],
                'uncertainty': predictions['uncertainty'],
                'confidence': predictions['confidence']
            }

            # Main multi-obstacle loss (without count)
            loss, components = improved_multi_obstacle_loss(predictions, obstacle_targets)

            # Add count classification loss
            count_loss = F.cross_entropy(count_preds, obstacle_counts_gt.squeeze(1).long())
            loss += count_loss
            components['count_loss'] = count_loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(resnet.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            for key, value in components.items():
                key_map = {
                    'standard_loss': 'standard',
                    'confidence_loss': 'confidence',
                    'gate_reg': 'gate_reg',
                    'uncertainty_reg': 'uncertainty_reg',
                    'count_loss': 'count'  # NEW
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
    total_obstacles = 0
    matched_obstacles = 0
    total_absolute_error = []

    pred_counts_all = []
    true_counts_all = []

    tolerance = 10  # degrees

    gate_stats = {
        'min_gate_avg': 0.0,
        'max_gate_avg': 0.0,
        'uncertainty_avg': 0.0,
        'confidence_avg': 0.0,
    }
    num_batches = 0

    for batch_idx, (distances, labels, _, tsdf) in enumerate(test_loader, start=1):
        gt_obstacle_targets = generate_multi_obstacle_targets(labels)

        distances = distances.to(device)
        tsdf = tsdf.to(device)

        # Get predictions
        predictions = resnet(distances, tsdf)

        min_pred_logits = predictions['min_logits']  # (B, 360)
        max_pred_logits = predictions['max_logits']  # (B, 360)
        count_preds = predictions['count']           # (B,) — now class indices, not logits

        gating_info = {
            'min_gate': predictions['min_gate'],
            'max_gate': predictions['max_gate'],
            'uncertainty': predictions['uncertainty'],
            'confidence': predictions['confidence']
        }

        batch_size = distances.size(0)
        total_samples += batch_size

        confidence = torch.sigmoid(gating_info['confidence']).detach().cpu()

        for b in range(batch_size):
            gt_spans = gt_obstacle_targets[b]
            conf_scores = confidence[b].numpy().squeeze()

            pred_count = int(torch.argmax(count_preds[b]).item())
            true_count = len(gt_spans)

            pred_counts_all.append(pred_count)
            true_counts_all.append(true_count)

            conf_peaks, _ = find_peaks(conf_scores, height=0.5, distance=5)
            sorted_peaks = conf_peaks[np.argsort(conf_scores[conf_peaks])[::-1]]

            pred_spans = []
            used_mask = np.zeros(360, dtype=bool)
            span_width = 10

            i = 0
            while len(pred_spans) < pred_count and i < len(sorted_peaks):
                center = sorted_peaks[i]
                i += 1

                min_idx = (center - span_width) % 360
                max_idx = (center + span_width) % 360

                if min_idx <= max_idx:
                    span_bins = list(range(min_idx, max_idx + 1))
                else:
                    span_bins = list(range(min_idx, 360)) + list(range(0, max_idx + 1))

                if any(used_mask[bin_idx] for bin_idx in span_bins):
                    continue

                for bin_idx in span_bins:
                    used_mask[bin_idx] = True

                pred_spans.append((min_idx, max_idx))

            while len(pred_spans) < pred_count:
                pred_spans.append((0, 0))

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

            if batch_idx == 8:
                print(f"Batch {batch_idx} Sample {b+1}/{batch_size} Predicted Spans: {pred_spans}")
                print(f"Batch {batch_idx} Sample {b+1}/{batch_size} Ground Truth Spans: {gt_spans}")
                print(f"Pred Count: {pred_count}, True Count: {true_count}")

        gate_stats['min_gate_avg'] += gating_info['min_gate'].mean().item()
        gate_stats['max_gate_avg'] += gating_info['max_gate'].mean().item()
        gate_stats['uncertainty_avg'] += gating_info['uncertainty'].mean().item()
        gate_stats['confidence_avg'] += confidence.mean().item()
        num_batches += 1

    avg_error = np.mean(total_absolute_error)
    obstacle_accuracy = 100.0 * matched_obstacles / total_obstacles

    pred_counts_all = np.array(pred_counts_all)
    true_counts_all = np.array(true_counts_all)
    count_mae = np.mean(np.abs(pred_counts_all - true_counts_all))
    exact_match = np.sum(pred_counts_all == true_counts_all)
    count_accuracy = 100.0 * exact_match / len(pred_counts_all)

    print(f'Total GT Obstacles: {total_obstacles}')
    print(f'Matched Obstacles (±{tolerance}° width): {matched_obstacles}')
    print(f'Obstacle Detection Accuracy: {obstacle_accuracy:.2f}%')
    print(f'Average Width Error: {avg_error:.2f} degrees')
    print(f'Count MAE: {count_mae:.2f}')
    print(f'Count Accuracy (Exact Match): {count_accuracy:.2f}%')

    print('Gating Statistics:')
    for k, v in gate_stats.items():
        print(f'  {k.replace("_", " ").title()}: {v / num_batches:.3f}')

    plt.figure(figsize=(15, 10))

    plt.subplot(2, 3, 1)
    plt.plot(range(num_epochs), loss_history, label='Total Loss')
    plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.title('Training Loss'); plt.legend()

    # plt.subplot(2, 3, 2)
    # for key, values in loss_components.items():
    #     if hasattr(values, 'cpu'):
    #         values = values.cpu().numpy()
    #         plt.plot(range(num_epochs), values, label=key)
    # plt.xlabel('Epochs'); plt.ylabel('Loss Component'); plt.title('Loss Components'); plt.legend()

    plt.subplot(2, 3, 3)
    gate_names = list(gate_stats.keys())
    gate_values = [v / num_batches for v in gate_stats.values()]
    plt.bar(gate_names, gate_values)
    plt.title('Gating Statistics'); plt.xticks(rotation=45)

    plt.subplot(2, 3, 4)
    plt.hist(np.abs(pred_counts_all - true_counts_all), bins=np.arange(0, max(6, pred_counts_all.max()) + 1), rwidth=0.8)
    plt.xlabel('Absolute Count Error'); plt.ylabel('Frequency')
    plt.title('Count Prediction Error Histogram')

    plt.tight_layout()
    plt.show()

    return obstacle_accuracy, avg_error, count_mae, count_accuracy, gate_stats







if __name__ == "__main__":
    # if not os.path.exists(os.path.join(data_dir, "tsdf_train.npy")):
    #     _ = lidar_to_tsdf_obstacles(training_lidar, img_dim=64)

    # if not os.path.exists(os.path.join(data_dir, "tsdf_test.npy")):
    #     _ = lidar_to_tsdf_obstacles(testing_lidar, img_dim=64)

    loss_history, loss_components = train_model()
    obstacle_accuracy, avg_error, count_mae, count_accuracy, gate_stats = test_model_with_gating(loss_history, loss_components)
    # tolerance_accuracy, avg_absolute_error, gate_stats = test_model_with_gating_loader(
    #     resnet, test_loader, generate_multi_obstacle_targets,
    #     loss_history, loss_components, num_epochs
    # )

    model_path = os.path.join(data_dir, "resnet_counting.pth")
    torch.save(resnet.state_dict(), model_path)
    print(f'Model saved to {model_path}')
    
    # Additional analysis
    print("\n=== GATING ANALYSIS ===")
    print(f"The model learned to use min_gate with weight {gate_stats['min_gate_avg']:.3f}")
    print(f"The model learned to use max_gate with weight {gate_stats['max_gate_avg']:.3f}")
    print(f"Average uncertainty: {gate_stats['uncertainty_avg']:.3f}")
    print(f"Average confidence: {gate_stats['confidence_avg']:.3f}")
