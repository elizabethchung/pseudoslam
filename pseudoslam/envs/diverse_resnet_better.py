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
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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
num_epochs = 50
batch_size = 32
learning_rate = 0.00005
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


def generate_multi_obstacle_targets(labels):
    """
    Generate obstacle angle regions (min, max) for each sample in the batch.
    Wraparound regions are merged into a single (start, end) pair where end < start.
    """
    batch_targets = []
    for sample_idx in range(labels.shape[0]):
        label_row = torch.tensor(labels[sample_idx], device='cuda' if torch.cuda.is_available() else 'cpu')
        device = label_row.device

        # Detect transitions using padding
        padded = torch.cat([torch.tensor([0], device=device), label_row, torch.tensor([0], device=device)])
        diff = torch.diff(padded)
        starts = torch.where(diff == 1)[0]
        ends = torch.where(diff == -1)[0] - 1

        obstacle_regions = list(zip(starts.tolist(), ends.tolist()))

        # Check for wraparound: last value is 1 and first is also 1
        if label_row[0] == 1 and label_row[-1] == 1:
            # Merge the first and last regions into a wraparound (e.g., (44, 1))
            if len(obstacle_regions) >= 2:
                first_start, first_end = obstacle_regions[0]
                last_start, last_end = obstacle_regions[-1]
                merged_region = (last_start, first_end)

                # Remove first and last, add merged
                obstacle_regions = obstacle_regions[1:-1]
                obstacle_regions.append(merged_region)

        # Print and save
        # if sample_idx == 0:
        #     print(f"Sample {sample_idx}: {len(obstacle_regions)} obstacles: {obstacle_regions}")
        batch_targets.append(obstacle_regions)

    return batch_targets

def extract_min_max_from_targets(batch_targets, device):
    """
    Converts list of obstacle tuples to two tensors: min_indices and max_indices.
    Pads with -1 where obstacles are missing.
    """
    max_obs = max(len(sample) for sample in batch_targets)
    min_indices = []
    max_indices = []

    for sample in batch_targets:
        mins = [t[0] for t in sample]
        maxs = [t[1] for t in sample]

        # Pad with -1 if fewer than max_obs
        mins += [-1] * (max_obs - len(mins))
        maxs += [-1] * (max_obs - len(maxs))

        min_indices.append(mins)
        max_indices.append(maxs)

    return (
        torch.tensor(min_indices, device=device, dtype=torch.long),
        torch.tensor(max_indices, device=device, dtype=torch.long),
    )

def improved_multi_obstacle_loss_with_constraints(predictions, obstacle_targets,
                                   alpha=1.0, confidence_weight=0.5,
                                   diversity_weight=0.1, max_ce_loss=10.0,
                                   empty_slot_penalty=0.05):
    """
    Enhanced loss function that encourages finding all obstacles
    """
    min_pred_logits = predictions['min_logits']  # (B, 3, 360)
    max_pred_logits = predictions['max_logits']  # (B, 3, 360)
    confidence_logits = predictions['confidence']  # (B, 3, 360)

    batch_size = min_pred_logits.shape[0]
    max_obstacles = min_pred_logits.shape[1]

    total_position_loss = 0.0
    total_confidence_loss = 0.0
    total_diversity_loss = 0.0
    total_completeness_loss = 0.0  # New: encourage using all slots
    num_position_losses = 0
    num_confidence_losses = 0
    num_diversity_pairs = 0

    for batch_idx in range(batch_size):
        obstacle_regions = obstacle_targets[batch_idx]
        num_obstacles = len(obstacle_regions) if obstacle_regions else 0

        confidence_probs = torch.sigmoid(confidence_logits[batch_idx])  # (max_obstacles, 360)

        # Diversity loss - encourage different slots to focus on different areas
        for i in range(max_obstacles):
            for j in range(i + 1, max_obstacles):
                inter = torch.sum(torch.min(confidence_probs[i], confidence_probs[j]))
                union = torch.sum(torch.max(confidence_probs[i], confidence_probs[j]))
                if union > 0:
                    iou = inter / union
                    total_diversity_loss += iou
                    num_diversity_pairs += 1

        # Completeness loss - encourage using all slots when there are obstacles
        if num_obstacles > 0:
            slot_usage = torch.sum(confidence_probs, dim=1)  # (max_obstacles,)
            # Encourage slots to be either fully used or not used at all
            for slot_idx in range(max_obstacles):
                if slot_idx < num_obstacles:
                    # Should be active - penalize low usage
                    usage_penalty = torch.relu(10.0 - slot_usage[slot_idx])
                    total_completeness_loss += usage_penalty
                else:
                    # Should be inactive - penalize high usage
                    usage_penalty = torch.relu(slot_usage[slot_idx] - 5.0)
                    total_completeness_loss += usage_penalty * empty_slot_penalty

        # Position and confidence losses
        for slot_idx in range(max_obstacles):
            if slot_idx < num_obstacles:
                region = obstacle_regions[slot_idx]
                min_class, max_class = region[0], region[1]

                # Position losses with adaptive weighting
                ce_min = F.cross_entropy(min_pred_logits[batch_idx, slot_idx:slot_idx+1],
                                       torch.tensor([min_class], device=min_pred_logits.device))
                ce_max = F.cross_entropy(max_pred_logits[batch_idx, slot_idx:slot_idx+1],
                                       torch.tensor([max_class], device=max_pred_logits.device))

                ce_min = torch.clamp(ce_min, 0, max_ce_loss)
                ce_max = torch.clamp(ce_max, 0, max_ce_loss)

                # Weight by obstacle size - smaller obstacles get higher weight
                obstacle_size = min(abs(max_class - min_class), 360 - abs(max_class - min_class))
                size_weight = 1.0 + (20.0 / max(obstacle_size, 5))  # Boost small obstacles

                total_position_loss += (ce_min + ce_max) * size_weight
                num_position_losses += 2

                # Confidence target
                confidence_target = torch.zeros_like(confidence_logits[batch_idx, slot_idx])
                if min_class <= max_class:
                    confidence_target[min_class:max_class + 1] = 1.0
                else:
                    confidence_target[min_class:] = 1.0
                    confidence_target[:max_class + 1] = 1.0

                conf_loss = F.binary_cross_entropy_with_logits(
                    confidence_logits[batch_idx, slot_idx],
                    confidence_target
                )
                conf_loss = torch.clamp(conf_loss, 0, max_ce_loss)
                total_confidence_loss += conf_loss
                num_confidence_losses += 1

            else:
                # Empty slot - light penalty only
                conf_loss = F.binary_cross_entropy_with_logits(
                    confidence_logits[batch_idx, slot_idx],
                    torch.zeros_like(confidence_logits[batch_idx, slot_idx])
                )
                conf_loss = torch.clamp(conf_loss, 0, max_ce_loss)
                total_confidence_loss += conf_loss * empty_slot_penalty
                num_confidence_losses += 1

    # Normalize losses
    avg_position_loss = total_position_loss / max(num_position_losses, 1)
    avg_confidence_loss = total_confidence_loss / max(num_confidence_losses, 1)
    avg_diversity_loss = total_diversity_loss / max(num_diversity_pairs, 1)
    avg_completeness_loss = total_completeness_loss / batch_size

    total_loss = (
        alpha * avg_position_loss +
        confidence_weight * avg_confidence_loss +
        diversity_weight * avg_diversity_loss +
        0.1 * avg_completeness_loss  # Completeness encouragement
    )

    return total_loss, {
        'position_loss': avg_position_loss.item(),
        'confidence_loss': avg_confidence_loss.item(),
        'diversity_loss': avg_diversity_loss.item(),
        'completeness_loss': avg_completeness_loss.item(),
        'num_position_losses': num_position_losses,
        'num_confidence_losses': num_confidence_losses,
        'num_diversity_pairs': num_diversity_pairs
    }


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
            targets = generate_multi_obstacle_targets(self.labels)  # list of [ [(min1, max1), (min2, max2), ...], ... ]

            # Extract min and max for each obstacle
            min_indices = []
            max_indices = []

            for sample in targets:
                mins = [t[0] for t in sample]
                maxs = [t[1] for t in sample]
                min_indices.append(mins)
                max_indices.append(maxs)

            self.min_indices = torch.tensor(min_indices, device=device)  # shape: [batch_size, num_obstacles]
            self.max_indices = torch.tensor(max_indices, device=device)  # same

        min_len = min(len(self.data), len(self.labels), len(self.mask) if self.mask is not None else float('inf'))
        if len(self.data) != len(self.labels) or (self.mask is not None and len(self.data) != len(self.mask)):
            print(f"Warning: data length ({len(self.data)}), labels length ({len(self.labels)}), mask length ({len(self.mask) if self.mask is not None else 'N/A'})")
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
    def __init__(self, in_channels, out_channels, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.conv3 = nn.Conv1d(out_channels, out_channels, 3, padding=1)
        self.bn3 = nn.BatchNorm1d(out_channels)
        self.shortcut = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.relu(self.bn3(self.conv3(out)))
        out = self.dropout(out)
        out = out + residual
        return self.relu(out)

class ImprovedResNetNoGating3(nn.Module):
    def __init__(self):
        super().__init__()
        self.max_obstacles = 3
       
        # More powerful LiDAR processing
        self.lidar_layer1 = ResidualBlock(1, 64, dropout=0.1)
        self.lidar_layer2 = ResidualBlock(64, 128, dropout=0.1)
        self.lidar_layer3 = ResidualBlock(128, 256, dropout=0.2)
        self.lidar_layer4 = ResidualBlock(256, 256, dropout=0.2)

        # Multi-head attention for LiDAR
        self.lidar_attention = nn.MultiheadAttention(embed_dim=256, num_heads=8, batch_first=True)
       
        # Enhanced TSDF processing
        self.tsdf_conv2d = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, None))
        )

        # Enhanced fusion
        self.fusion_layer = nn.Sequential(
            nn.Conv1d(512, 512, kernel_size=5, padding=2),  # 256 + 256
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        self.dropout_fused = nn.Dropout(p=0.1)
       
        # Shared feature extractor
        self.shared_features = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
       
        # Task-specific heads
        self.min_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(64, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Conv1d(32, 1, 1)
            ) for _ in range(self.max_obstacles)
        ])
       
        self.max_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(64, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Conv1d(32, 1, 1)
            ) for _ in range(self.max_obstacles)
        ])
       
        self.confidence_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(64, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Dropout(0.15),
                nn.Conv1d(32, 1, 1)
            ) for _ in range(self.max_obstacles)
        ])
       
        self._initialize_heads_with_diversity()
   
    def _initialize_heads_with_diversity(self):
        """Initialize heads to encourage diversity"""
        for i, head in enumerate(self.min_heads):
            for layer in head:
                if isinstance(layer, nn.Conv1d):
                    nn.init.xavier_normal_(layer.weight, gain=0.8)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, -0.3 - i * 0.2)
       
        for i, head in enumerate(self.max_heads):
            for layer in head:
                if isinstance(layer, nn.Conv1d):
                    nn.init.xavier_normal_(layer.weight, gain=0.8)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0.3 + i * 0.2)
       
        for i, head in enumerate(self.confidence_heads):
            for layer in head:
                if isinstance(layer, nn.Conv1d):
                    nn.init.xavier_normal_(layer.weight, gain=0.5)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, -2.0 + i * 0.5)  # Different starting confidences
   
    def forward(self, lidar_data, tsdf_data):
        batch_size = lidar_data.shape[0]
       
        # Enhanced LiDAR processing
        x_lidar = self.lidar_layer1(lidar_data)
        x_lidar = self.lidar_layer2(x_lidar)
        x_lidar = self.lidar_layer3(x_lidar)
        x_lidar = self.lidar_layer4(x_lidar)
       
        # Apply attention
        x_lidar_transposed = x_lidar.transpose(1, 2)  # (B, 360, 256)
        x_lidar_attended, _ = self.lidar_attention(x_lidar_transposed, x_lidar_transposed, x_lidar_transposed)
        x_lidar = x_lidar_attended.transpose(1, 2)  # Back to (B, 256, 360)

        # Process TSDF
        while tsdf_data.ndim > 4:
            tsdf_data = tsdf_data.squeeze(0)
       
        if tsdf_data.shape[-1] != 360:
            tsdf_data = F.interpolate(tsdf_data, size=(tsdf_data.shape[2], 360),
                                    mode='bilinear', align_corners=False)

        tsdf_feat = self.tsdf_conv2d(tsdf_data)  # (B, 256, 1, 360)
        tsdf_feat = tsdf_feat.squeeze(2)  # (B, 256, 360)

        # Ensure same sequence length
        if tsdf_feat.shape[-1] != x_lidar.shape[-1]:
            tsdf_feat = F.interpolate(tsdf_feat, size=x_lidar.shape[-1],
                                    mode='linear', align_corners=False)
       
        # Fusion
        x_fused = torch.cat([x_lidar, tsdf_feat], dim=1)  # (B, 512, 360)
        x_fused = self.fusion_layer(x_fused)  # (B, 128, 360)
        x_fused = self.dropout_fused(x_fused)

        # Shared features
        shared_features = self.shared_features(x_fused)  # (B, 64, 360)

        # Generate predictions
        all_min_logits = []
        all_max_logits = []
        all_confidence = []
       
        for i in range(self.max_obstacles):
            min_logits = self.min_heads[i](shared_features).squeeze(1)  # (B, 360)
            max_logits = self.max_heads[i](shared_features).squeeze(1)  # (B, 360)
            confidence = self.confidence_heads[i](shared_features).squeeze(1)  # (B, 360)
           
            all_min_logits.append(min_logits)
            all_max_logits.append(max_logits)
            all_confidence.append(confidence)

        return {
            'min_logits': torch.stack(all_min_logits, dim=1),  # (B, 3, 360)
            'max_logits': torch.stack(all_max_logits, dim=1),  # (B, 3, 360)
            'confidence': torch.stack(all_confidence, dim=1)  # (B, 3, 360)
        }
    

# Load data
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, "data")
training_lidar = np.load(os.path.join(data_dir, "diverse_training_lidar_3.npy"))
training_labels = np.load(os.path.join(data_dir, "diverse_training_labels_3.npy"))
testing_lidar = np.load(os.path.join(data_dir, "diverse_testing_lidar_3.npy"))
labels_test = np.load(os.path.join(data_dir, "diverse_testing_labels_3.npy"))


# Debug: Print shapes to verify
print(f"Training lidar shape: {training_lidar.shape}")
print(f"Training labels shape: {training_labels.shape}")
print(f"Testing lidar shape: {testing_lidar.shape}")
print(f"Testing labels shape: {labels_test.shape}")

# Use labels as tensors
training_targets = generate_multi_obstacle_targets(training_labels)
training_min_indices, training_max_indices = extract_min_max_from_targets(training_targets, device)
testing_targets = generate_multi_obstacle_targets(labels_test)
testing_min_indices, testing_max_indices = extract_min_max_from_targets(testing_targets, device)


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

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=8)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=8)


# Use improved model without gating
resnet = ImprovedResNetNoGating3().to(device)


def train_model():
    model = ImprovedResNetNoGating3().to(device)
   
    # Use a more sophisticated optimizer setup
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)
   
    # Warm-up + cosine annealing scheduler
    def lr_lambda(epoch):
        if epoch < 5:  # Warm-up
            return (epoch + 1) / 5
        else:  # Cosine annealing
            return 0.5 * (1 + math.cos(math.pi * (epoch - 5) / (num_epochs - 5)))
   
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
   
    loss_history = []
    loss_components = {'position_loss': [], 'confidence_loss': [],
                      'diversity_loss': [], 'completeness_loss': []}
   
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_components = {key: 0.0 for key in loss_components.keys()}
        batch_count = 0
       
        for batch_idx, (distances, labels, _, tsdf) in enumerate(train_loader):
            distances = distances.to(device)
            tsdf = tsdf.to(device)
           
            obstacle_targets = generate_multi_obstacle_targets(labels)
           
            # Skip batches with no obstacles during early training
            if epoch < 5 and not any(len(targets) > 0 for targets in obstacle_targets):
                continue

            optimizer.zero_grad()
           
            predictions = model(distances, tsdf)
           
            # Use the enhanced loss function
            loss, components = improved_multi_obstacle_loss_with_constraints(
                predictions, obstacle_targets,
                alpha=1.0, confidence_weight=0.6, diversity_weight=0.15,
                empty_slot_penalty=0.1
            )
           
            if torch.isnan(loss):
                print(f"NaN loss at epoch {epoch}, batch {batch_idx}")
                continue
           
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
           
            epoch_loss += loss.item()
            for key in epoch_components:
                if key in components:
                    epoch_components[key] += components[key]
            batch_count += 1
           
            if batch_idx % 20 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], '
                      f'Loss: {loss.item():.4f}')
       
        scheduler.step()
       
        if batch_count == 0:
            continue
       
        avg_epoch_loss = epoch_loss / batch_count
        loss_history.append(avg_epoch_loss)
       
        for key in epoch_components:
            loss_components[key].append(epoch_components[key] / batch_count)
       
        print(f'Epoch [{epoch+1}/{num_epochs}] Average Loss: {avg_epoch_loss:.4f}')
        print(f' Position: {epoch_components["position_loss"]/batch_count:.4f}, '
              f'Confidence: {epoch_components["confidence_loss"]/batch_count:.4f}, '
              f'Diversity: {epoch_components["diversity_loss"]/batch_count:.4f}, '
              f'Completeness: {epoch_components["completeness_loss"]/batch_count:.4f}')
   
    return model, loss_history, loss_components

def improved_post_processing(predictions, confidence_threshold=0.3,
                           min_obstacle_width=1, max_obstacle_width=200,
                           num_expected_obstacles=3):
    """
    More sophisticated post-processing that better handles multiple obstacles
    """
    min_pred_logits = predictions['min_logits']  # (B, num_slots, 360)
    max_pred_logits = predictions['max_logits']  # (B, num_slots, 360)
    confidence_logits = predictions['confidence']  # (B, num_slots, 360)

    batch_size = min_pred_logits.shape[0]
    num_slots = min_pred_logits.shape[1]

    batch_results = []

    for b in range(batch_size):
        confidence_probs = torch.sigmoid(confidence_logits[b]).detach().cpu().numpy()  # (num_slots, 360)
        min_probs = torch.softmax(min_pred_logits[b], dim=-1).detach().cpu().numpy()  # (num_slots, 360)
        max_probs = torch.softmax(max_pred_logits[b], dim=-1).detach().cpu().numpy()

        obstacle_candidates = []

        for slot_idx in range(num_slots):
            # Check if this slot is confident about detecting an obstacle
            max_confidence = np.max(confidence_probs[slot_idx])
            avg_confidence = np.mean(confidence_probs[slot_idx])
           
            if max_confidence < confidence_threshold and avg_confidence < confidence_threshold * 0.5:
                continue  # Skip slots with low confidence

            # Find confident regions
            confident_indices = np.where(confidence_probs[slot_idx] > confidence_threshold)[0]
           
            if len(confident_indices) < min_obstacle_width:
                continue

            # Find the best min/max within confident regions
            # Weight the probabilities by confidence
            weighted_min_probs = min_probs[slot_idx] * confidence_probs[slot_idx]
            weighted_max_probs = max_probs[slot_idx] * confidence_probs[slot_idx]

            best_min_idx = np.argmax(weighted_min_probs)
            best_max_idx = np.argmax(weighted_max_probs)

            # Ensure reasonable span
            if best_min_idx == best_max_idx:
                # Expand around the point
                best_min_idx = (best_min_idx - 2) % 360
                best_max_idx = (best_max_idx + 2) % 360

            # Calculate span width
            if best_min_idx <= best_max_idx:
                span_width = best_max_idx - best_min_idx + 1
            else:
                span_width = (360 - best_min_idx) + (best_max_idx + 1)

            # Filter by reasonable span width
            if span_width < min_obstacle_width or span_width > max_obstacle_width:
                continue

            # Calculate combined score
            min_confidence_at_pred = confidence_probs[slot_idx, best_min_idx]
            max_confidence_at_pred = confidence_probs[slot_idx, best_max_idx]
            position_certainty = (min_probs[slot_idx, best_min_idx] + max_probs[slot_idx, best_max_idx]) / 2
           
            combined_score = (min_confidence_at_pred + max_confidence_at_pred + position_certainty) / 3

            obstacle_candidates.append({
                'min_idx': best_min_idx,
                'max_idx': best_max_idx,
                'score': combined_score,
                'span_width': span_width,
                'slot': slot_idx,
                'avg_confidence': avg_confidence
            })

        # Sort by score (higher is better)
        obstacle_candidates.sort(key=lambda x: x['score'], reverse=True)

        # Select non-overlapping obstacles using a more sophisticated method
        selected_obstacles = []
        used_regions = []

        for candidate in obstacle_candidates:
            # Create the span
            if candidate['min_idx'] <= candidate['max_idx']:
                candidate_span = set(range(candidate['min_idx'], candidate['max_idx'] + 1))
            else:
                candidate_span = set(range(candidate['min_idx'], 360)) | set(range(0, candidate['max_idx'] + 1))

            # Check overlap with existing selections
            overlap_detected = False
            for existing_span in used_regions:
                overlap = len(candidate_span & existing_span)
                overlap_ratio = overlap / len(candidate_span)
               
                # Allow small overlaps but not major ones
                if overlap_ratio > 0.3:  # More than 30% overlap
                    overlap_detected = True
                    break

            if not overlap_detected:
                selected_obstacles.append((candidate['min_idx'], candidate['max_idx']))
                used_regions.append(candidate_span)

                if len(selected_obstacles) >= num_expected_obstacles:
                    break

        # If we didn't find enough obstacles, add the best remaining candidates
        # This ensures we always return the expected number
        remaining_candidates = [c for c in obstacle_candidates
                              if (c['min_idx'], c['max_idx']) not in selected_obstacles]
       
        while len(selected_obstacles) < num_expected_obstacles and remaining_candidates:
            best_remaining = remaining_candidates.pop(0)
            selected_obstacles.append((best_remaining['min_idx'], best_remaining['max_idx']))

        # Final fallback - add dummy obstacles if still not enough
        while len(selected_obstacles) < num_expected_obstacles:
            # Place dummy obstacles in unused space
            dummy_start = (len(selected_obstacles) * 60) % 360
            dummy_end = (dummy_start + 10) % 360
            selected_obstacles.append((dummy_start, dummy_end))

        batch_results.append(selected_obstacles)

    return batch_results



def test_single_scan(distances, tsdf, num_expected_obstacles=3):
    """
    Run a single unlabeled LiDAR scan through the model and return predicted obstacle spans.
    """
    resnet.eval()

    if isinstance(distances, np.ndarray):
        distances = torch.tensor(distances, dtype=torch.float32)
    if distances.ndim == 1: 
        distances = distances.unsqueeze(0)  # (1, 360)
    if distances.ndim == 2:
        distances = distances.unsqueeze(1)  # (1, 1, 360)

    distances = distances.to(device)

    if tsdf is not None:
        if isinstance(tsdf, np.ndarray):
            tsdf = torch.tensor(tsdf, dtype=torch.float32)
        if tsdf.ndim == 3:
            tsdf = tsdf.unsqueeze(0)  # (1, C, H, W)
        tsdf = tsdf.to(device)
    else:
        # Dummy TSDF if not provided — adjust shape if needed
        tsdf = torch.zeros((1, 1, 30, 30), device=device)

    with torch.no_grad():
        predictions = resnet(distances, tsdf)
        spans = improved_post_processing(predictions, num_expected_obstacles=num_expected_obstacles)

    return spans[0]


def test_model_with_improved_processing(loss_history, loss_components):
    """
    Updated test function using the improved post-processing
    """
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
        
        # Use improved post-processing
        batch_predicted_spans = improved_post_processing(predictions, num_expected_obstacles=3)

        batch_size = distances.size(0)
        total_samples += batch_size

        for b in range(batch_size):
            gt_spans = gt_obstacle_targets[b]
            pred_spans = batch_predicted_spans[b]

            # Debug output for some samples
            if num_batches <= 3 and b < 5:
                print(f"Batch {num_batches}, Sample {b+1}:")
                print(f"  Predicted Spans: {pred_spans}")
                print(f"  Ground Truth Spans: {gt_spans}")

            total_obstacles += len(gt_spans)

            # Match predicted spans to ground truth
            for gt_min, gt_max in gt_spans:
                gt_width = (gt_max - gt_min + 360) % 360 + 1
                best_error = float('inf')
                
                for pred_min, pred_max in pred_spans:
                    pred_width = (pred_max - pred_min + 360) % 360 + 1
                    error = abs(pred_width - gt_width)
                    best_error = min(best_error, error)
                
                total_absolute_error.append(best_error)
                if best_error <= tolerance:
                    matched_obstacles += 1

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
        # Skip if empty
        if values is None or len(values) == 0:
            continue

        # Convert tensors to CPU + numpy
        if hasattr(values, 'cpu'):
            values = values.cpu().numpy()
        elif isinstance(values, list):
            values = [v.cpu().item() if hasattr(v, 'cpu') else v for v in values]

        # Skip if still wrong shape
        if len(values) != num_epochs:
            print(f"Skipping '{key}' due to mismatched length: expected {num_epochs}, got {len(values)}")
            continue

        plt.plot(range(num_epochs), values, label=key)

    plt.xlabel('Epochs')
    plt.ylabel('Loss Component')
    plt.title('Loss Components')
    plt.legend()


    plt.subplot(2, 3, 3)
    plt.title('No Gating Statistics')
    plt.text(0.5, 0.5, 'Gating removed', ha='center', va='center', fontsize=14)
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    return obstacle_accuracy, avg_error


def test_model_with_improved_processing_single(dataloader, num_expected_obstacles=3, verbose=False):
    """
    Runs the model on the provided dataloader and returns predicted obstacle spans.

    Args:
        dataloader (DataLoader): PyTorch DataLoader containing (distances, labels, _, tsdf)
        num_expected_obstacles (int): Number of predicted obstacle spans per sample
        verbose (bool): Whether to print debug output for first few batches

    Returns:
        all_predictions (List[List[Tuple[int, int]]]): List of predicted spans for each sample
    """
    test_resnet = ImprovedResNetNoGating3().to(device)
    model_path = os.path.join(data_dir, "diverse_resnet_better.pth")
    test_resnet.load_state_dict(torch.load(model_path))
    test_resnet.eval()
    all_predictions = []
    num_batches = 0

    for batch_idx, (distances, labels, _, tsdf) in enumerate(dataloader, start=1):
        distances = distances.to(device)
        tsdf = tsdf.to(device)

        with torch.no_grad():
            predictions = test_resnet(distances, tsdf)

        batch_predicted_spans = improved_post_processing(predictions, num_expected_obstacles=num_expected_obstacles)
        all_predictions.extend(batch_predicted_spans)

        if verbose and num_batches <= 3:
            for b, pred_spans in enumerate(batch_predicted_spans):
                print(f"Batch {num_batches}, Sample {b+1}: Predicted Spans: {pred_spans}")

        num_batches += 1

    return all_predictions

def test_model_with_improved_processing_multi(dataloader, num_expected_obstacles=3, verbose=False, model_path=None):
    """
    Runs the model on the provided dataloader and returns predicted obstacle spans for all samples.

    Args:
        dataloader (DataLoader): PyTorch DataLoader containing (distances, labels, _, tsdf)
        num_expected_obstacles (int): Number of predicted obstacle spans per sample
        verbose (bool): Whether to print debug output for first few batches
        model_path (str, optional): Path to the model weights. If None, uses default.

    Returns:
        all_predictions (List[List[Tuple[int, int]]]): List of predicted spans for each sample in the dataloader
    """
    test_resnet = ImprovedResNetNoGating3().to(device)
    if model_path is None:
        model_path = os.path.join(data_dir, "diverse_resnet_better.pth")
    test_resnet.load_state_dict(torch.load(model_path, map_location=device))
    test_resnet.eval()
    all_predictions = []
    num_batches = 0

    for batch_idx, (distances, labels, _, tsdf) in enumerate(dataloader, start=1):
        distances = distances.to(device)
        tsdf = tsdf.to(device)

        with torch.no_grad():
            predictions = test_resnet(distances, tsdf)

        batch_predicted_spans = improved_post_processing(predictions, num_expected_obstacles=num_expected_obstacles)
        all_predictions.extend(batch_predicted_spans)

        if verbose and num_batches < 3:
            for b, pred_spans in enumerate(batch_predicted_spans):
                print(f"Batch {num_batches}, Sample {b+1}: Predicted Spans: {pred_spans}")

        num_batches += 1

    return all_predictions

def inspect_dataset_constructor():
    """
    Helper to understand your LidarTSDFDataset constructor
    """
    import inspect
    sig = inspect.signature(LidarTSDFDataset.__init__)
    print("LidarTSDFDataset constructor signature:")
    for param_name, param in sig.parameters.items():
        if param_name != 'self':
            print(f"  {param_name}: {param}")

# Run this first to see what parameters your dataset expects
inspect_dataset_constructor()


if __name__ == "__main__":
    # train_model_few_batches()
    model, loss_history, loss_components = train_model()
    obstacle_accuracy, avg_error = test_model_with_improved_processing(loss_history, loss_components)
    # sample_lidar = testing_lidar[200]  # Use the first sample for testing
    # sample_tsdf = lidar_to_tsdf_obstacles(np.expand_dims(sample_lidar, axis=0))  # Get corresponding TSDF
    # Generate dummy labels (all 0s)
    # dummy_labels = np.zeros((1, 360), dtype=np.float32)  # shape: (1, 360)
    # # dummy_labels = np.expand_dims(labels_test[200], axis=0)  # shape: (1, 360)
    # actual_obstacle_spans = generate_multi_obstacle_targets(np.expand_dims(labels_test[200], axis=0))  # Get ground truth
    # single_test = LidarTSDFDataset(
    #     np.expand_dims(sample_lidar, axis=0), 
    #     dummy_labels,  # Dummy labels
    #     np.expand_dims(testing_mask[200], axis=0),
    #     tsdf_cache_path=None)

    # single_test = DataLoader(single_test, batch_size=1, shuffle=False, pin_memory=True, num_workers=4)

    # predicted_spans = test_model_with_improved_processing_single(single_test)

    # print("Predicted obstacle spans:")
    # print(predicted_spans)
    # print("Actual obstacle spans:")
    # print(actual_obstacle_spans)

    model_path = os.path.join(data_dir, "diverse_resnet_better.pth")
    torch.save(resnet.state_dict(), model_path)
    print(f'Model saved to {model_path}')

