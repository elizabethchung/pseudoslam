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

def multi_span_loss(pred_spans, pred_confs, target_spans, gating_info, lambda_overlap=10.0):
    """
    pred_spans: (B, N, 2)
    pred_confs: (B, N)
    target_spans: list of lists of (min_angle, max_angle)
    """
    batch_size, N, _ = pred_spans.shape
    total_loss = 0.0
    mse_loss = nn.MSELoss(reduction='mean')
    overlap_penalty = 0.0
    matched_count = 0

    for b in range(batch_size):
        pred = pred_spans[b]    # (N, 2)
        conf = pred_confs[b]    # (N,)
        target = target_spans[b]  # list of (min_angle, max_angle)

        sample_loss = 0.0
        matches = min(len(target), N)

        for i in range(matches):
            target_span = torch.tensor(target[i], dtype=torch.float32, device=pred.device)
            sample_loss += mse_loss(pred[i], target_span)
            matched_count += 1

        total_loss += sample_loss

        # Penalty for overlapping predicted spans
        for i in range(N):
            for j in range(i+1, N):
                min_i, max_i = pred[i]
                min_j, max_j = pred[j]

                inter = torch.clamp(torch.min(max_i, max_j) - torch.max(min_i, min_j), min=0)
                overlap_penalty += inter

    # Normalize
    if matched_count > 0:
        total_loss /= matched_count

    total_loss += lambda_overlap * overlap_penalty

    return total_loss, {
        'span_mse': total_loss.item() - lambda_overlap * overlap_penalty.item(),
        'overlap_penalty': overlap_penalty.item(),
        'total_loss': total_loss.item()
    }



def loss(pred_spans, pred_confs, target_spans, gating_info, lambda_overlap=10.0):
    """
    pred_spans: (B, N, 2)
    pred_confs: (B, N)
    target_spans: list of lists of (min_angle, max_angle)
    """
    batch_size, N, _ = pred_spans.shape
    total_loss = 0.0
    mse_loss = nn.MSELoss(reduction='mean')
    overlap_penalty = 0.0
    matched_count = 0

    for b in range(batch_size):
        pred = pred_spans[b]    # (N, 2)
        conf = pred_confs[b]    # (N,)
        target = target_spans[b]  # list of (min_angle, max_angle)

        sample_loss = 0.0
        matches = min(len(target), N)

        for i in range(matches):
            target_span = torch.tensor(target[i], dtype=torch.float32, device=pred.device)
            sample_loss += mse_loss(pred[i], target_span)
            matched_count += 1

        total_loss += sample_loss

        # Penalty for overlapping predicted spans
        for i in range(N):
            for j in range(i+1, N):
                min_i, max_i = pred[i]
                min_j, max_j = pred[j]

                inter = torch.clamp(torch.min(max_i, max_j) - torch.max(min_i, min_j), min=0)
                overlap_penalty += inter

    # Normalize
    if matched_count > 0:
        total_loss /= matched_count

    total_loss += lambda_overlap * overlap_penalty

    return total_loss, {
        'span_mse': total_loss.item() - lambda_overlap * overlap_penalty.item(),
        'overlap_penalty': overlap_penalty.item(),
        'total_loss': total_loss.item()
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

class SingleGatingHead(nn.Module):
    def __init__(self, feature_dim=64, hidden_dim=32):
        super().__init__()
        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(feature_dim, hidden_dim, 1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, 1),
            nn.ReLU()
        )
        self.local_context = nn.Sequential(
            nn.Conv1d(feature_dim, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU()
        )
        self.gate_network = nn.Sequential(
            nn.Conv1d(hidden_dim * 2, hidden_dim, 1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, 2, 1),  # min and max gates
        )
        self.confidence_network = nn.Sequential(
            nn.Conv1d(hidden_dim * 2, hidden_dim, 1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, features):
        batch_size, _, seq_len = features.shape
        global_ctx = self.global_context(features).expand(-1, -1, seq_len)
        local_ctx = self.local_context(features)
        combined = torch.cat([global_ctx, local_ctx], dim=1)  # (B, 2H, 360)

        # Gating over min/max
        gate_logits = self.gate_network(combined)  # (B, 2, 360)
        gate_weights = F.softmax(gate_logits, dim=1)  # (B, 2, 360)
        min_gate = gate_weights[:, 0, :]  # (B, 360)
        max_gate = gate_weights[:, 1, :]  # (B, 360)

        # Confidence
        conf_map = self.confidence_network(combined)  # (B, 1, 360)
        confidence = F.adaptive_avg_pool1d(conf_map, 1).squeeze(-1).squeeze(-1)  # (B,)

        # Angle values
        angle_vals = torch.linspace(0, 359, seq_len, device=features.device).view(1, -1)  # (1, 360)
        min_angle = torch.sum(min_gate * angle_vals, dim=1)  # (B,)
        max_angle = torch.sum(max_gate * angle_vals, dim=1)  # (B,)

        return min_angle, max_angle, confidence, {
            'min_gate': min_gate.unsqueeze(1),  # (B, 1, 360)
            'max_gate': max_gate.unsqueeze(1),  # (B, 1, 360)
            'uncertainty': torch.abs(min_gate - max_gate).unsqueeze(1)  # (B, 1, 360)
        }


class MultiSpanGatingModule(nn.Module):
    def __init__(self, feature_dim=64, hidden_dim=32, max_spans=3):
        super().__init__()
        self.heads = nn.ModuleList([
            SingleGatingHead(feature_dim, hidden_dim)
            for _ in range(max_spans)
        ])
        self.max_spans = max_spans

    def forward(self, features):
        """
        Args:
            features: Tensor of shape (B, C, 360)
        Returns:
            spans: Tensor of shape (B, N, 2) — min/max angles per span
            confidences: Tensor of shape (B, N)
        """
        span_preds = []
        confs = []
        gates_list = []

        for head in self.heads:
            min_angle, max_angle, conf, gates = head(features)
            span_preds.append(torch.stack([min_angle, max_angle], dim=-1))  # (B, 2)
            confs.append(conf)  # (B,)
            gates_list.append(gates)

        spans = torch.stack(span_preds, dim=1)       # (B, N, 2)
        confidences = torch.stack(confs, dim=1)      # (B, N)

        return spans, confidences, {
            'min_gate': torch.cat([g['min_gate'] for g in gates_list], dim=1),  # (B, N, 360)
            'max_gate': torch.cat([g['max_gate'] for g in gates_list], dim=1),  # (B, N, 360)
            'uncertainty': torch.cat([g['uncertainty'] for g in gates_list], dim=1)  # (B, N, 360)
        }


class Resnet(nn.Module):
    def __init__(self, max_spans=3):
        super().__init__()
        self.max_spans = max_spans

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

        # Fusion layers
        self.fusion_conv1 = nn.Conv1d(192, 128, kernel_size=3, padding=1)
        self.fusion_conv2 = nn.Conv1d(128, 64, kernel_size=3, padding=1)

        # Gating module (now multi-span)
        self.gating_module = MultiSpanGatingModule(feature_dim=64, hidden_dim=32, max_spans=max_spans)
        self.min_gate = None
        self.max_gate = None
        self.uncertainty = None
        self.confidence = None

        self.dropout = nn.Dropout(0.1)

    def forward(self, lidar_data, tsdf_data):
        # LiDAR path
        x_lidar = self.lidar_layer1(lidar_data)
        x_lidar = self.lidar_layer2(x_lidar)
        x_lidar = self.lidar_layer3(x_lidar)

        # TSDF pre-processing
        while tsdf_data.ndim > 4:
            tsdf_data = tsdf_data.squeeze(0)
        assert tsdf_data.ndim == 4, f"Expected 4D TSDF, got {tsdf_data.shape}"

        if tsdf_data.shape[-1] != 360:
            tsdf_data = F.interpolate(tsdf_data, size=(tsdf_data.shape[2], 360), mode='bilinear', align_corners=False)

        tsdf_radial = tsdf_data.mean(dim=2)
        assert tsdf_radial.shape[1] == 2, f"TSDF channels should be 2, got {tsdf_radial.shape[1]}"

        # TSDF path
        tsdf_feat = self.tsdf_conv1d(tsdf_radial)

        # Feature fusion
        x_fused = torch.cat([x_lidar, tsdf_feat], dim=1)
        x_fused = torch.relu(self.fusion_conv1(x_fused))
        x_fused = torch.relu(self.fusion_conv2(x_fused))
        x_fused = self.dropout(x_fused)

        

        # Gating prediction (returns span start, end, confidence per span)
        spans, confidences, gates = self.gating_module(x_fused)  # spans: (B, N, 2), confidences: (B, N)

        self.min_gate = gates['min_gate']
        self.max_gate = gates['max_gate']
        self.uncertainty = gates['uncertainty']
        self.confidence = confidences

        return spans, confidences, {
            'min_gate': self.min_gate,
            'max_gate': self.max_gate,
            'uncertainty': self.uncertainty,
            'confidence': confidences
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
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # pseudoslam/envs (data/ lives there)
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
resnet = Resnet().to(device)

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
            pred_spans, confidences, gating_info = resnet(distances, tsdf)

            loss_val, components = loss(pred_spans, confidences, obstacle_targets, gating_info)

            
            loss_val.backward()
            torch.nn.utils.clip_grad_norm_(resnet.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss_val.item()
            
            # If you want to track loss components, update here after modifying the loss function to return them.
            # Currently, only total loss is tracked.
            
            for key in components:
                if key in epoch_components:
                    epoch_components[key] += components[key]
                else:
                    epoch_components[key] = components[key]

            if batch_idx % 1 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], '
                      f'Loss: {loss_val.item():.4f}')
        
        scheduler.step()
        avg_epoch_loss = epoch_loss / len(train_loader)
        loss_history.append(avg_epoch_loss)
        
        for key in epoch_components:
            loss_components[key].append(epoch_components[key] / len(train_loader))
        
        print(f'Epoch [{epoch+1}/{num_epochs}] Average Loss: {avg_epoch_loss:.4f}')
    
    return loss_history, loss_components

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
        gt_obstacle_targets = generate_multi_obstacle_targets(labels)

        distances = distances.to(device)
        tsdf = tsdf.to(device)

        if batch_idx % 1 == 0:
            # Print first sample's labels and generated targets for debugging
            print(f'Batch {batch_idx} Sample 10 Scan:', distances[9].cpu().numpy())
            print(f"Batch {batch_idx} Sample 10 Labels:", labels[9])
            print(f"Batch {batch_idx} Sample 10 GT Obstacle Targets:", gt_obstacle_targets[9])

        # Forward pass with gating info
        spans, confidences, gating_info = resnet(distances, tsdf)

        pred_min_scores = torch.sigmoid(spans[:, :, 0]).cpu().detach().numpy()
        pred_max_scores = torch.sigmoid(spans[:, :, 1]).cpu().detach().numpy()

        batch_size = distances.size(0)
        total_samples += batch_size

        for b in range(batch_size):
            gt_spans = gt_obstacle_targets[b]

            min_peaks, _ = find_peaks(pred_min_scores[b], height=peak_threshold)
            max_peaks, _ = find_peaks(pred_max_scores[b], height=peak_threshold)

            pred_spans = []
            for min_p in min_peaks:
                max_candidates = max_peaks[max_peaks >= min_p]
                if len(max_candidates) == 0 and len(max_peaks) > 0:
                    max_p = max_peaks[0]
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
                    best_error = min(best_error, error)

                total_absolute_error.append(best_error)
                if best_error <= tolerance:
                    matched_obstacles += 1

            if batch_idx == 7:
                print(f"Batch {batch_idx} Sample {b+1}/{batch_size} Predicted Spans: {pred_spans}")
                print(f"Batch {batch_idx} Sample {b+1}/{batch_size} Ground Truth Spans: {gt_spans}")

        # Gating stat aggregation
        gate_stats['min_gate_avg'] += gating_info['min_gate'].mean().item()
        gate_stats['max_gate_avg'] += gating_info['max_gate'].mean().item()
        gate_stats['uncertainty_avg'] += gating_info['uncertainty'].mean().item()
        gate_stats['confidence_avg'] += gating_info['confidence'].mean().item()
        num_batches += 1

    avg_error = np.mean(total_absolute_error)
    obstacle_accuracy = 100.0 * matched_obstacles / total_obstacles

    print(f'\nEvaluation Results:')
    print(f'Total GT Obstacles: {total_obstacles}')
    print(f'Matched Obstacles (±{tolerance}° width): {matched_obstacles}')
    print(f'Obstacle Detection Accuracy: {obstacle_accuracy:.2f}%')
    print(f'Average Width Error: {avg_error:.2f}°\n')

    print(f'Gating Averages:')
    for k in gate_stats:
        print(f'  {k.replace("_", " ").title()}: {gate_stats[k] / num_batches:.3f}')

    # Plot results
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 3, 1)
    plt.plot(range(len(loss_history)), loss_history, label='Total Loss')
    plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.title('Training Loss')
    plt.legend()

    plt.subplot(2, 3, 2)
    for key, values in loss_components.items():
        plt.plot(range(len(values)), values, label=key)
    plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.title('Loss Components')
    plt.legend()

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
    
    model_path = os.path.join(data_dir, "gated_resnet_multi.pth")
    torch.save(resnet.state_dict(), model_path)
    print(f'Model saved to {model_path}')
    
    # Additional analysis
    print("\n=== GATING ANALYSIS ===")
    print(f"The model learned to use min_gate with weight {gate_stats['min_gate_avg']:.3f}")
    print(f"The model learned to use max_gate with weight {gate_stats['max_gate_avg']:.3f}")
    print(f"Average uncertainty: {gate_stats['uncertainty_avg']:.3f}")
    print(f"Average confidence: {gate_stats['confidence_avg']:.3f}")
