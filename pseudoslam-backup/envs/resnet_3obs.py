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
import math

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

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


def lidar_to_tsdf_obstacles(lidar_data, img_dim=32, max_range=200.0, trunc_thresh=3):
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
        device = label_row.device  # Get device of the input
        diff = torch.diff(torch.cat([torch.tensor([0], device=device), label_row, torch.tensor([0], device=device)]))
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
                                    alpha=0.5,
                                    confidence_weight=1.0):
    """
    Gradient-safe loss function that avoids in-place operations
    """
    
    min_pred_logits = predictions['min_logits']  # (B, 3, 360)
    max_pred_logits = predictions['max_logits']  # (B, 3, 360)
    confidence_logits = predictions['confidence']  # (B, 3, 360)
    
    batch_size = min_pred_logits.shape[0]
    max_obstacles = min_pred_logits.shape[1]
    
    # Initialize loss components
    total_loss = torch.tensor(0.0, device=min_pred_logits.device, requires_grad=True)
    position_loss_sum = torch.tensor(0.0, device=min_pred_logits.device)
    confidence_loss_sum = torch.tensor(0.0, device=min_pred_logits.device)
    
    for batch_idx in range(batch_size):
        obstacle_regions = obstacle_targets[batch_idx]
        num_obstacles = len(obstacle_regions) if obstacle_regions else 0
        
        # Create confidence targets - AVOID in-place operations!
        confidence_targets = torch.zeros_like(confidence_logits[batch_idx])  # (3, 360)
        
        batch_position_loss = torch.tensor(0.0, device=min_pred_logits.device)
        batch_confidence_loss = torch.tensor(0.0, device=min_pred_logits.device)
        
        # Process each obstacle slot
        for slot_idx in range(max_obstacles):
            if slot_idx < num_obstacles:
                region = obstacle_regions[slot_idx]
                
                # Ensure valid indices
                num_classes = min_pred_logits.shape[2]  # 360
                min_class = int(max(0, min(region[0], num_classes - 1)))
                max_class = int(max(0, min(region[1], num_classes - 1)))
                
                # Position loss
                min_target = torch.tensor(min_class, dtype=torch.long, device=min_pred_logits.device)
                max_target = torch.tensor(max_class, dtype=torch.long, device=max_pred_logits.device)

                # Print min_pred_logits[batch_idx, slot_idx:slot_idx+1], min_target.unsqueeze(0)
                # print(f"min_pred_logits: {min_pred_logits[batch_idx, slot_idx:slot_idx+1]}, min_target: {min_target.unsqueeze(0)}")
                # Print shape of logits
                # print(f"min_pred_logits shape: {min_pred_logits.shape}, min_target shape: {min_target.unsqueeze(0).shape}")

                assert min_target.dtype == torch.long, f"min_target wrong dtype: {min_target.dtype}"
                assert 0 <= min_target.item() < 360, f"min_target out of bounds: {min_target.item()}"
                assert not torch.isnan(min_pred_logits[batch_idx, slot_idx]).any(), "NaN in logits"

                # Shape of logits is (B, 3, 360), shape of targets is (1,)
                ce_min = F.cross_entropy(min_pred_logits[batch_idx, slot_idx:slot_idx+1], min_target.unsqueeze(0))
                ce_max = F.cross_entropy(max_pred_logits[batch_idx, slot_idx:slot_idx+1], max_target.unsqueeze(0))
                
                batch_position_loss = batch_position_loss + ce_min + ce_max

                # Check if model is learning anything
                print(f"Loss value: {ce_min.item():.4f}")
                print(f"Random baseline: {math.log(360):.4f}")

                # Check prediction diversity
                preds = torch.argmax(min_pred_logits[batch_idx, slot_idx], dim=-1)
                print(f"Unique predictions: {torch.unique(preds).numel()}/360")

                # Check target distribution
                print(f"Target value: {min_target.item()}")
                print(f"Most confident prediction: {torch.argmax(min_pred_logits[batch_idx, slot_idx]).item()}")
                
                # CRITICAL FIX: Create new tensor instead of in-place modification
                # OLD (BROKEN): confidence_targets[slot_idx, min_class:max_class + 1] = 1.0
                
                # Create a mask and use it to set targets
                target_mask = torch.zeros_like(confidence_targets[slot_idx])
                if min_class <= max_class:
                    target_mask[min_class:max_class + 1] = 1.0
                else:
                    # Handle wrap-around
                    target_mask[min_class:] = 1.0
                    target_mask[:max_class + 1] = 1.0
                
                # Use torch.where to avoid in-place operation
                slot_target = torch.zeros_like(confidence_logits[batch_idx, slot_idx])

                if min_class <= max_class:
                    slot_target[min_class:max_class + 1] = 1.0
                else:
                    slot_target[min_class:] = 1.0
                    slot_target[:max_class + 1] = 1.0

                # Accumulate confidence target without in-place ops
                confidence_targets = confidence_targets.clone()  # detach from graph to be safe
                confidence_targets[slot_idx] = slot_target

            
            # Confidence loss for this slot
            slot_conf_loss = F.binary_cross_entropy_with_logits(
                confidence_logits[batch_idx, slot_idx], 
                confidence_targets[slot_idx]
            )
            batch_confidence_loss = batch_confidence_loss + slot_conf_loss
        
        # Accumulate losses
        position_loss_sum = position_loss_sum + batch_position_loss
        confidence_loss_sum = confidence_loss_sum + batch_confidence_loss
    
    # Normalize and combine losses
    avg_position_loss = position_loss_sum / batch_size
    avg_confidence_loss = confidence_loss_sum / batch_size
    total_loss = alpha * avg_position_loss + confidence_weight * avg_confidence_loss
    
    components = {
        'position_loss': avg_position_loss.item(),
        'confidence_loss': avg_confidence_loss.item()
    }
    
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

class ImprovedResNetNoGating(nn.Module):
    def __init__(self):
        super().__init__()
        self.max_obstacles = 3  # Hardcoded obstacle count
        
        # LiDAR processing
        self.lidar_layer1 = ResidualBlock(1, 16)
        self.lidar_layer2 = ResidualBlock(16, 32)
        self.lidar_layer3 = ResidualBlock(32, 64)
        
        # TSDF Conv2D path
        self.tsdf_conv2d = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # Fusion layers
        self.fusion_conv1 = nn.Conv1d(96, 128, kernel_size=3, padding=1)
        self.fusion_conv2 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
        
        self.obstacle_heads = nn.ModuleList([
            nn.ModuleDict({
                'min_head': nn.Sequential(
                    nn.Conv1d(64, 32, 1),
                    nn.ReLU(),
                    nn.Dropout(0.1),  # Add regularization
                    nn.Conv1d(32, 16, 1),  # Different depth
                    nn.ReLU(),
                    nn.Conv1d(16, 1, 1)
                ),
                'max_head': nn.Sequential(
                    nn.Conv1d(64, 48, 1),  # Different width
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Conv1d(48, 24, 1),
                    nn.ReLU(),
                    nn.Conv1d(24, 1, 1)
                )
            }) for _ in range(self.max_obstacles)
        ])
        
        # FIXED: Better confidence head with proper initialization
        self.confidence_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(64, 32, 1),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Conv1d(32, 16, 1),
                nn.ReLU(),
                nn.Conv1d(16, 1, 1)
            ) for _ in range(self.max_obstacles)
        ])
        
        # CRITICAL: Different initialization for min/max heads
        self._initialize_heads()
        
        self.dropout = nn.Dropout(0.1)
        
    def _initialize_heads(self):
        """Initialize min/max heads differently to avoid identical behavior"""
        for i, obstacle_head in enumerate(self.obstacle_heads):
            # Initialize min_head with negative bias (prefer smaller angles)
            for layer in obstacle_head['min_head']:
                if isinstance(layer, nn.Conv1d):
                    nn.init.xavier_normal_(layer.weight)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, -0.1)  # Negative bias
            
            # Initialize max_head with positive bias (prefer larger angles)
            for layer in obstacle_head['max_head']:
                if isinstance(layer, nn.Conv1d):
                    nn.init.xavier_normal_(layer.weight)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0.1)   # Positive bias
        
        # Initialize confidence heads to output ~0.1 initially (conservative)
        for conf_head in self.confidence_heads:
            for layer in conf_head:
                if isinstance(layer, nn.Conv1d):
                    nn.init.xavier_normal_(layer.weight)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, -2.0)  # sigmoid(-2) ≈ 0.12
                
    def forward(self, lidar_data, tsdf_data):
        """Fixed forward pass without in-place operations"""
        
        # Process LiDAR
        x_lidar = self.lidar_layer1(lidar_data)
        x_lidar = self.lidar_layer2(x_lidar)
        x_lidar = self.lidar_layer3(x_lidar)

        # Process TSDF
        while tsdf_data.ndim > 4:
            tsdf_data = tsdf_data.squeeze(0)
        assert tsdf_data.ndim == 4, f"Expected 4D TSDF, got {tsdf_data.shape}"

        if tsdf_data.shape[-1] != 360:
            tsdf_data = F.interpolate(tsdf_data, size=(tsdf_data.shape[2], 360), mode='bilinear', align_corners=False)

        tsdf_feat = self.tsdf_conv2d(tsdf_data)
        tsdf_feat = F.adaptive_avg_pool2d(tsdf_feat, (1, 360))
        tsdf_feat = tsdf_feat.squeeze(2)

        if tsdf_feat.shape[-1] != x_lidar.shape[-1]:
            tsdf_feat = F.interpolate(tsdf_feat, size=x_lidar.shape[-1], mode='linear', align_corners=False)
        
        x_fused = torch.cat([x_lidar, tsdf_feat], dim=1)
        x_fused = torch.relu(self.fusion_conv1(x_fused))
        x_fused = torch.relu(self.fusion_conv2(x_fused))
        x_fused = self.dropout(x_fused)

        # FIXED: Avoid in-place operations
        all_min_logits = []
        all_max_logits = []
        all_confidence = []
        
        for i in range(self.max_obstacles):
            min_logits = self.obstacle_heads[i]['min_head'](x_fused).squeeze(1)
            max_logits = self.obstacle_heads[i]['max_head'](x_fused).squeeze(1)
            confidence = self.confidence_heads[i](x_fused).squeeze(1)
            
            all_min_logits.append(min_logits)
            all_max_logits.append(max_logits)
            all_confidence.append(confidence)

        # CRITICAL FIX: Use torch.stack instead of any in-place operations
        stacked_min = torch.stack(all_min_logits, dim=1)      # (B, 3, 360)
        stacked_max = torch.stack(all_max_logits, dim=1)      # (B, 3, 360)
        stacked_confidence = torch.stack(all_confidence, dim=1)  # (B, 3, 360)

        # Print shape of outputs for debugging
        print(f"Output shapes: min_logits={stacked_min.shape}, max_logits={stacked_max.shape}, confidence={stacked_confidence.shape}")

        return {
            'min_logits': stacked_min,
            'max_logits': stacked_max,
            'confidence': stacked_confidence
        }
    

# Load data
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, "data")
training_lidar = np.load(os.path.join(data_dir, "fix_training_lidar_3obs.npy"))[:100]
training_labels = np.load(os.path.join(data_dir, "fix_training_labels_3obs.npy"))[:100]
testing_lidar = np.load(os.path.join(data_dir, "fix_testing_lidar_3obs.npy"))[:20]
labels_test = np.load(os.path.join(data_dir, "fix_testing_labels_3obs.npy"))[:20]


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

print("Batch min/max labels:", training_labels.min().item(), training_labels.max().item())


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
    tsdf_cache_path=os.path.join(data_dir, "tsdf_train_3obs.npy")
)

test_dataset = LidarTSDFDataset(
    testing_lidar, 
    (testing_min_indices, testing_max_indices), 
    testing_mask,
    tsdf_cache_path=os.path.join(data_dir, "tsdf_test_3obs.npy")
)

# debug_dataset_labels(train_dataset, num_samples=5)
# debug_dataset_labels(test_dataset, num_samples=5)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)


# Use improved model without gating
resnet = ImprovedResNetNoGating().to(device)

def train_model_one_batch():
    optimizer = optim.Adam(resnet.parameters(), lr=0.0005, weight_decay=1e-4)

    # Freeze one batch from the train loader
    batch = next(iter(train_loader))
    distances, labels, _, tsdf = batch
    distances = distances.to(device)
    labels = labels.to(device)
    tsdf = tsdf.to(device)

    # Prepare static target for this batch
    obstacle_targets = generate_multi_obstacle_targets(labels)
    print("Targets (sample):", obstacle_targets[0])


    resnet.train()

    for epoch in range(200):
        optimizer.zero_grad()

        # Forward pass
        predictions = resnet(distances, tsdf)  # unpack: min_logits, max_logits, confidence

        # Loss computation
        loss, components = improved_multi_obstacle_loss(predictions, obstacle_targets)

        # Backprop + update
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")


def train_model():
    optimizer = optim.Adam(resnet.parameters(), lr=0.0005, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    loss_history = []
    loss_components = {'position_loss': [], 'confidence_loss': []}
    
    for epoch in range(num_epochs):
        resnet.train()
        epoch_loss = 0.0
        epoch_components = {'position_loss': 0.0, 'confidence_loss': 0.0}
        
        for batch_idx, (distances, labels, _, tsdf) in enumerate(train_loader):
            distances = distances.to(device)
            tsdf = tsdf.to(device)
            
            # Generate multi-obstacle targets
            obstacle_targets = generate_multi_obstacle_targets(labels)

            print("Targets (sample):", obstacle_targets[0])

            optimizer.zero_grad()
            predictions = resnet(distances, tsdf)

            # Use multi-obstacle loss (no gating)
            loss, components = improved_multi_obstacle_loss(predictions, obstacle_targets)

            loss.backward()

            # # === Print gradients for debug ===
            # print("Gradient debug info:")
            # for name, param in resnet.named_parameters():
            #     if param.grad is not None:
            #         print(f"  {name}: grad mean={param.grad.mean().item():.6f}, grad std={param.grad.std().item():.6f}")
            #     else:
            #         print(f"  {name}: grad=None")

            torch.nn.utils.clip_grad_norm_(resnet.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            for key in epoch_components:
                if key in components:
                    epoch_components[key] += components[key]
            
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


def test_model_no_gating(loss_history, loss_components):
    resnet.eval()
    total_samples = 0
    total_obstacles = 0
    matched_obstacles = 0
    total_absolute_error = []

    tolerance = 5  # degrees
    num_batches = 0

    for batch_idx, (distances, labels, _, tsdf) in enumerate(test_loader, start=1):
        gt_obstacle_targets = generate_multi_obstacle_targets(labels)

        distances = distances.to(device)
        tsdf = tsdf.to(device)

        predictions = resnet(distances, tsdf)
        min_pred_logits = predictions['min_logits']  # (B, 3, 360)
        max_pred_logits = predictions['max_logits']  # (B, 3, 360)
        confidence = torch.sigmoid(predictions['confidence']).detach().cpu()

        batch_size = distances.size(0)
        total_samples += batch_size

        for b in range(batch_size):
            gt_spans = gt_obstacle_targets[b]
            conf_scores_2d = confidence[b].numpy()  # (3, 360)
            min_logits_2d = min_pred_logits[b].detach().cpu().numpy()  # (3, 360)
            max_logits_2d = max_pred_logits[b].detach().cpu().numpy()  # (3, 360)

            pred_spans = []
            used_bins = np.zeros(360, dtype=bool)

            for obs_idx in range(3):
                conf_scores = conf_scores_2d[obs_idx]
                min_logits = min_logits_2d[obs_idx]
                max_logits = max_logits_2d[obs_idx]

                peak = np.argmax(conf_scores)

                search_range = [(peak + i) % 360 for i in range(-30, 31)]

                min_idx = max(search_range, key=lambda i: min_logits[i])
                max_idx = max(search_range, key=lambda i: max_logits[i])

                if (max_idx - min_idx + 360) % 360 < 1:
                    print(f"[WARN] Skipped span too narrow: obs {obs_idx}, sample {b}, batch {batch_idx}")
                    continue

                span_bins = list(range(min_idx, max_idx + 1)) if min_idx <= max_idx else \
                            list(range(min_idx, 360)) + list(range(0, max_idx + 1))
                if any(used_bins[i] for i in span_bins):
                    print(f"[WARN] Skipped span overlapping: obs {obs_idx}, sample {b}, batch {batch_idx}")
                    continue

                for i in span_bins:
                    used_bins[i] = True

                pred_spans.append((min_idx, max_idx))

            # Fallback if < 3 spans
            if len(pred_spans) < 3:
                print(f"[INFO] Fallback triggered: {3 - len(pred_spans)} spans missing for sample {b}, batch {batch_idx}")
                fallback_peaks = np.argsort(conf_scores_2d.mean(axis=0))[::-1]
                for peak in fallback_peaks:
                    if len(pred_spans) >= 3:
                        break
                    if used_bins[peak]:
                        continue

                    min_idx = (peak - 5) % 360
                    max_idx = (peak + 5) % 360

                    span_bins = list(range(min_idx, max_idx + 1)) if min_idx <= max_idx else \
                                list(range(min_idx, 360)) + list(range(0, max_idx + 1))

                    if any(used_bins[i] for i in span_bins):
                        continue

                    for i in span_bins:
                        used_bins[i] = True

                    pred_spans.append((min_idx, max_idx))

            # Final hard fallback
            while len(pred_spans) < 3:
                pred_spans.append((0, 10))

            # Step 4: Scoring
            total_obstacles += len(gt_spans)

            for gt_min, gt_max in gt_spans:
                gt_width = (gt_max - gt_min + 360) % 360
                best_error = float('inf')
                for pred_min, pred_max in pred_spans:
                    pred_width = (pred_max - pred_min + 360) % 360
                    error = abs(pred_width - gt_width)
                    best_error = min(best_error, error)
                total_absolute_error.append(best_error)
                if best_error <= tolerance:
                    matched_obstacles += 1

            if num_batches % 2 == 0:
                print(f"Batch {num_batches}, Sample {b+1}:")
                print(f"  Predicted Spans: {pred_spans}")
                print(f"  Ground Truth Spans: {gt_spans}")

        num_batches += 1

    avg_error = np.mean(total_absolute_error)
    obstacle_accuracy = 100.0 * matched_obstacles / total_obstacles

    print(f'Total GT Obstacles: {total_obstacles}')
    print(f'Matched Obstacles (±{tolerance}° width): {matched_obstacles}')
    print(f'Obstacle Detection Accuracy: {obstacle_accuracy:.2f}%')
    print(f'Average Width Error: {avg_error:.2f} degrees')

    plt.figure(figsize=(15, 10))

    plt.subplot(2, 3, 1)
    plt.plot(range(num_epochs), loss_history, label='Total Loss')
    plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.title('Training Loss'); plt.legend()

    plt.subplot(2, 3, 2)
    for key, values in loss_components.items():
        if hasattr(values, 'cpu'):
            values = values.cpu().numpy()
        elif isinstance(values, list) and len(values) > 0 and hasattr(values[0], 'cpu'):
            values = [v.cpu().item() if hasattr(v, 'cpu') else v for v in values]
        plt.plot(range(num_epochs), values, label=key)
    plt.xlabel('Epochs'); plt.ylabel('Loss Component'); plt.title('Loss Components'); plt.legend()

    plt.subplot(2, 3, 3)
    plt.title('No Gating Statistics')
    plt.text(0.5, 0.5, 'Gating removed', ha='center', va='center', fontsize=14)
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    return obstacle_accuracy, avg_error





if __name__ == "__main__":
    

    loss_history, loss_components = train_model()
    obstacle_accuracy, avg_error = test_model_no_gating(loss_history, loss_components)


    model_path = os.path.join(data_dir, "resnet_3.pth")
    torch.save(resnet.state_dict(), model_path)
    print(f'Model saved to {model_path}')

