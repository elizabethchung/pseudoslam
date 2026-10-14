import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
import os

# Use the same device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Training parameters
num_epochs = 50
batch_size = 32  # Reduced for more stable gradients
learning_rate = 0.0001  # Lower learning rate
torch.backends.cudnn.benchmark = True

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
                _, first_end = obstacle_regions[0]
                last_start, _ = obstacle_regions[-1]
                merged_region = (last_start, first_end)
                # Remove first and last, add merged
                obstacle_regions = obstacle_regions[1:-1]
                obstacle_regions.append(merged_region)

        # Print and save
        if sample_idx == 0:
            print(f"Sample {sample_idx}: {len(obstacle_regions)} obstacles: {obstacle_regions}")
        batch_targets.append(obstacle_regions)

    return batch_targets

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

class CountOnlyDataset(Dataset):
    """Simplified dataset for count prediction only"""
    def __init__(self, data, labels, mask=None, tsdf_cache_path=None):
        self.data = data
        self.mask = mask if mask is not None else np.ones((len(data), 360))

        # Handle labels format
        if isinstance(labels, tuple):
            self.min_indices, self.max_indices = labels
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

        # Compute obstacle counts
        obstacle_regions = generate_multi_obstacle_targets(self.labels)
        self.counts = np.array([len(regions) for regions in obstacle_regions])  # counts per sample as a 1D array

        # Handle length mismatch
        min_len = min(len(self.data), len(self.labels))
        if len(self.data) != len(self.labels):
            print(f"Warning: data length ({len(self.data)}) != labels length ({len(self.labels)})")
            print(f"Truncating to minimum length: {min_len}")
            self.data = self.data[:min_len]
            self.labels = self.labels[:min_len]
            self.counts = self.counts[:min_len]
            if self.mask is not None:
                self.mask = self.mask[:min_len]

        # TSDF computation/loading
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
        count = int(self.counts[idx])  # Ensure it's a plain Python int
        tsdf_tensor = torch.from_numpy(self.tsdf_maps[idx]).float()

        distances = torch.tensor(lidar_data, dtype=torch.float32).unsqueeze(0)  # (1, 360)
        count_tensor = torch.tensor(count, dtype=torch.long)  # scalar, not wrapped in a list
        mask_tensor = torch.tensor(self.mask[idx], dtype=torch.float32)  # (360,)
        
        return distances, count_tensor, mask_tensor, tsdf_tensor


class ResidualBlock(nn.Module):
    """Same as original"""
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

class CountOnlyNetwork(nn.Module):
    """Simplified network that only predicts obstacle count"""
    def __init__(self, max_count=3):
        super().__init__()
        self.max_count = max_count
        
        # LiDAR processing pathway with batch norm
        self.lidar_layer1 = ResidualBlock(1, 16)
        self.lidar_layer2 = ResidualBlock(16, 32)
        self.lidar_layer3 = ResidualBlock(32, 64)
        
        # TSDF 2D processing with batch norm
        self.tsdf_conv2d = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        # Fusion layers with batch norm
        self.fusion_conv1 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.fusion_conv2 = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        
        # Multiple pooling strategies for robustness
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        
        # Simpler, more stable count prediction head
        self.count_head = nn.Sequential(
            nn.Linear(128, 64),  # 64*2 from avg+max pooling
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, max_count + 1)  # 4 output classes (0, 1, 2, 3 obstacles)
        )
        
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, lidar_data, tsdf_data):
        # Process LiDAR
        x_lidar = self.lidar_layer1(lidar_data)    # (B, 16, 360)
        x_lidar = self.lidar_layer2(x_lidar)       # (B, 32, 360)
        x_lidar = self.lidar_layer3(x_lidar)       # (B, 64, 360)
        
        # Process TSDF
        while tsdf_data.ndim > 4:
            tsdf_data = tsdf_data.squeeze(0)
        assert tsdf_data.ndim == 4, f"Expected 4D TSDF, got {tsdf_data.shape}"
        
        # Ensure TSDF has 360 angular resolution
        if tsdf_data.shape[-1] != 360:
            tsdf_data = F.interpolate(tsdf_data, size=(tsdf_data.shape[2], 360), mode='bilinear', align_corners=False)
        
        tsdf_feat = self.tsdf_conv2d(tsdf_data)  # (B, 64, H, 360)
        # Global average pooling across spatial dimensions, keep angular
        tsdf_feat = F.adaptive_avg_pool2d(tsdf_feat, (1, 360))  # (B, 64, 1, 360)
        tsdf_feat = tsdf_feat.squeeze(2)  # (B, 64, 360)
        
        # Fuse features
        x_fused = torch.cat([x_lidar, tsdf_feat], dim=1)  # (B, 128, 360)
        x_fused = self.fusion_conv1(x_fused)  # (B, 128, 360)
        x_fused = self.fusion_conv2(x_fused)  # (B, 64, 360)
        x_fused = self.dropout(x_fused)
        
        # Extract robust global features using both avg and max pooling
        global_avg = self.global_avg_pool(x_fused).squeeze(2)  # (B, 64)
        global_max = self.global_max_pool(x_fused).squeeze(2)  # (B, 64)
        global_feat = torch.cat([global_avg, global_max], dim=1)  # (B, 128)
        
        # Final classification
        count_logits = self.count_head(global_feat)  # (B, max_count+1)
        
        return count_logits

def create_biased_mask(labels, keep_1_prob=1.0, keep_0_prob=0.1):
    """Same as original"""
    mask = np.zeros_like(labels, dtype=np.float32)
    
    ones = (labels == 1)
    zeros = (labels == 0)

    mask[ones] = np.random.binomial(1, keep_1_prob, size=ones.sum())
    mask[zeros] = np.random.binomial(1, keep_0_prob, size=zeros.sum())
    
    return mask

def effective_number_weighting(class_counts, beta=0.999):
    """
    Computes weights using Effective Number of Samples formula.
    Args:
        class_counts: List or array of class counts.
        beta: Smoothing parameter (0.9–0.999 recommended).
    Returns:
        torch.FloatTensor of normalized class weights.
    """
    class_counts = np.array(class_counts)
    effective_num = 1.0 - np.power(beta, class_counts)
    weights = (1.0 - beta) / effective_num
    weights = weights / np.sum(weights) * len(class_counts)  # normalize
    return torch.tensor(weights, dtype=torch.float)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs, targets):
        log_probs = F.log_softmax(inputs, dim=1)              # (B, C)
        probs = torch.exp(log_probs)                          # (B, C)
        targets_one_hot = F.one_hot(targets, num_classes=inputs.size(1)).float()

        focal_weight = (1 - probs) ** self.gamma              # (B, C)
        loss = -targets_one_hot * focal_weight * log_probs    # (B, C)
        loss = loss.sum(dim=1)                                # (B,)

        if self.weight is not None:
            weight_per_sample = self.weight[targets]          # (B,)
            loss = loss * weight_per_sample

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


def train_count_model(model, train_loader, test_loader, num_epochs=50):
    """Train count model using Focal Loss with Effective Number class weighting"""
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.7, patience=5)

    # Effective class weights
    raw_counts = [1101, 493, 400, 66]
    raw_counts = [1101, 493, 400, 66]
    weights = effective_number_weighting(raw_counts, beta=0.999).to(device)
    print("Class weights:", weights)
    # Multiply the weights for rare classes (e.g., class 3) by a factor of 2 or 3
    weights[3] *= 3
    weights[2] *= 1.5
    weights = weights / weights.sum() * len(weights)

    criterion = FocalLoss(gamma=2.0, weight=weights)

    train_losses = []
    train_accuracies = []
    test_accuracies = []
    best_test_acc = 0.0
    patience_counter = 0
    early_stop_patience = 15

    for distances, counts, _, tsdf in train_loader:
        print(f"counts type: {type(counts)}, shape: {getattr(counts, 'shape', None)}")
        print(f"counts[:5]: {counts[:5]}")
        break


    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        for batch_idx, (distances, counts, _, tsdf) in enumerate(train_loader):
            distances = distances.to(device)
            counts = counts.to(device)
            tsdf = tsdf.to(device)

            # Ensure counts are in valid range [0, model.max_count-1]
            valid_mask = (counts >= 0) & (counts < model.max_count + 1)
            valid_mask = (counts >= 0) & (counts < model.max_count + 1)
            if not torch.all(valid_mask):
                print(f"Invalid count values detected in batch {batch_idx}: {counts}")
                
                # Find indices of invalid counts
                invalid_indices = (~valid_mask).nonzero(as_tuple=True)[0]
                
                # Print the lidar data corresponding to invalid counts
                for idx in invalid_indices:
                    print(f"Invalid sample index in batch: {idx.item()}")
                    print(f"Lidar data for invalid sample:\n{distances[idx]}")
                    
                
                raise ValueError(f"Counts must be in range [0, {model.max_count}]")

            optimizer.zero_grad()
            count_logits = model(distances, tsdf)

            loss = criterion(count_logits, counts.squeeze())  # Squeeze to remove any extra dimension
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

            epoch_loss += loss.item()
            predictions = torch.argmax(count_logits, dim=1)
            correct_predictions += (predictions == counts).sum().item()
            total_predictions += counts.size(0)
    
            if batch_idx % 50 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], '
                      f'Loss: {loss.item():.4f}')

        avg_train_loss = epoch_loss / len(train_loader)
        train_accuracy = 100.0 * float(correct_predictions) / float(total_predictions)

        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)

        test_accuracy = evaluate_count_model(model, test_loader, verbose=False)
        test_accuracies.append(test_accuracy)
        scheduler.step(test_accuracy)

        if test_accuracy > best_test_acc:
            best_test_acc = test_accuracy
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(data_dir, "best_count_model.pth"))
        else:
            patience_counter += 1

        print(f'Epoch [{epoch+1}/{num_epochs}] - Loss: {avg_train_loss:.4f}, '
              f'Train Acc: {train_accuracy:.2f}%, Test Acc: {test_accuracy:.2f}% (Best: {best_test_acc:.2f}%)')

        if patience_counter >= early_stop_patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

    return train_losses, train_accuracies, test_accuracies


def evaluate_count_model(model, test_loader, verbose=True):
    """Evaluate count prediction accuracy"""
    model.eval()
    correct_predictions = 0
    total_predictions = 0
    predictions_list = []
    ground_truth_list = []
    
    with torch.no_grad():
        for distances, counts, _, tsdf in test_loader:
            distances = distances.to(device)
            counts = counts.to(device)
            tsdf = tsdf.to(device)
            
            count_logits = model(distances, tsdf)
            predictions = torch.argmax(count_logits, dim=1)
            
            correct_predictions += (predictions == counts).sum().item()
            total_predictions += counts.size(0)
            
            predictions_list.extend(predictions.cpu().numpy())
            ground_truth_list.extend(counts.cpu().numpy())
    
    accuracy = 100.0 * correct_predictions / total_predictions
    
    if verbose:
        # Additional metrics
        predictions_array = np.array(predictions_list)
        ground_truth_array = np.array(ground_truth_list)
        
        mae = np.mean(np.abs(predictions_array - ground_truth_array))
        
        print(f"\nDetailed Results:")
        print(f"Total samples: {total_predictions}")
        print(f"Correct predictions: {correct_predictions}")
        print(f"Accuracy: {accuracy:.2f}%")
        print(f"Mean Absolute Error: {mae:.3f}")
        
        # Confusion matrix-like analysis
        print(f"\nCount distribution:")
        unique_true, counts_true = np.unique(ground_truth_array, return_counts=True)
        unique_pred, counts_pred = np.unique(predictions_array, return_counts=True)
        
        print("True counts:", dict(zip(unique_true, counts_true)))
        print("Predicted counts:", dict(zip(unique_pred, counts_pred)))
        
        # Per-class accuracy
        print("\nPer-class accuracy:")
        for count_val in range(5):  # 0-4 obstacles
            mask = ground_truth_array == count_val
            if mask.sum() > 0:
                class_acc = (predictions_array[mask] == count_val).mean() * 100
                print(f"  {count_val} obstacles: {class_acc:.1f}% ({mask.sum()} samples)")
    
    return accuracy

def plot_results(train_losses, train_accuracies, test_accuracies):
    """Plot training results"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss plot
    ax1.plot(train_losses)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Cross Entropy Loss')
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(train_accuracies, label='Train Accuracy')
    ax2.plot(test_accuracies, label='Test Accuracy')
    ax2.set_title('Accuracy Over Time')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Load data (same as original)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # pseudoslam/envs (data/ lives there)
    data_dir = os.path.join(base_dir, "data")
    
    training_lidar = np.load(os.path.join(data_dir, "fix_training_lidar.npy"))
    training_labels = np.load(os.path.join(data_dir, "fix_training_labels.npy"))
    training_lidar_2obs = np.load(os.path.join(data_dir, "fix_training_lidar_2obs.npy"))
    training_labels_2obs = np.load(os.path.join(data_dir, "fix_training_labels_2obs.npy"))
    training_lidar_3obs = np.load(os.path.join(data_dir, "fix_training_lidar_3obs.npy"))
    training_labels_3obs = np.load(os.path.join(data_dir, "fix_training_labels_3obs.npy"))
    # Make combined training data with extra 2/3 obstacle samples
    training_lidar = np.concatenate([training_lidar, training_lidar_2obs[:500], training_lidar_3obs[:250]], axis=0)
    training_labels = np.concatenate([training_labels, training_labels_2obs[:500], training_labels_3obs[:250]], axis=0)
    testing_lidar = np.load(os.path.join(data_dir, "fix_testing_lidar.npy"))
    labels_test = np.load(os.path.join(data_dir, "fix_testing_labels.npy"))
    testing_lidar = np.concatenate([testing_lidar, training_lidar_3obs[500:550]], axis=0)
    labels_test = np.concatenate([labels_test, training_labels_3obs[500:550]], axis=0)

    print("Checking obstacle counts in training labels...")
    obstacle_regions = generate_multi_obstacle_targets(training_labels)
    obstacle_counts = np.array([len(regions) for regions in obstacle_regions])  # counts per sample as a 1D array

    # Print count distribution
    unique_counts, counts_freq = np.unique(obstacle_counts, return_counts=True)
    print("Obstacle counts distribution in training labels:")
    for c, freq in zip(unique_counts, counts_freq):
        print(f"Count {c}: {freq} samples")

    # Find samples with obstacle count > 3
    bad_samples_idx = np.where(obstacle_counts > 3)[0]
    print(f"Number of training samples with obstacle count > 3: {len(bad_samples_idx)}")

    # Inspect a few of these samples
    for idx in bad_samples_idx[:10]:  # print up to 10 samples
        print(f"\nSample index {idx} has {obstacle_counts[idx]} obstacles")
        print(f"Label vector (first 50 elements): {training_labels[idx][:50]}")

    # Optionally: inspect label sums or unique values for suspicious samples
    for idx in bad_samples_idx[:10]:
        unique_vals = np.unique(training_labels[idx])
        print(f"Unique label values for sample {idx}: {unique_vals}")

    print(f"Training lidar shape: {training_lidar.shape}")
    print(f"Training labels shape: {training_labels.shape}")
    print(f"Testing lidar shape: {testing_lidar.shape}")
    print(f"Testing labels shape: {labels_test.shape}")
    
    # Convert to min/max format
    training_min_indices, training_max_indices = labels_to_min_max(training_labels)
    testing_min_indices, testing_max_indices = labels_to_min_max(labels_test)
    
    # Create masks
    training_mask = create_biased_mask(training_labels, keep_1_prob=1.0, keep_0_prob=0.1)
    testing_mask = np.ones_like(labels_test, dtype=np.float32)
    
    # Create datasets
    train_dataset = CountOnlyDataset(
        training_lidar, 
        (training_min_indices, training_max_indices), 
        training_mask,
        tsdf_cache_path=os.path.join(data_dir, "fix_tsdf_train.npy")
    )
    
    test_dataset = CountOnlyDataset(
        testing_lidar, 
        (testing_min_indices, testing_max_indices), 
        testing_mask,
        tsdf_cache_path=os.path.join(data_dir, "fix_tsdf_test.npy")
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)
    
    # Print count distributions
    print("\nCount distributions:")
    train_obstacle_regions = generate_multi_obstacle_targets(training_labels)
    train_counts = np.array([len(regions) for regions in train_obstacle_regions])  # counts per sample as a 1D array

    unique_train, counts_train = np.unique(train_counts, return_counts=True)
    print("Training counts:", dict(zip(unique_train, counts_train)))

    test_obstacle_regions = generate_multi_obstacle_targets(labels_test)
    test_counts = np.array([len(regions) for regions in test_obstacle_regions])  # counts per sample as a 1D array
    unique_test, counts_test = np.unique(test_counts, return_counts=True)
    print("Testing counts:", dict(zip(unique_test, counts_test)))
    
    # Create and train model
    model = CountOnlyNetwork(max_count=3).to(device)
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Train the model
    train_losses, train_accuracies, test_accuracies = train_count_model(
        model, train_loader, test_loader, num_epochs
    )
    
    # Final evaluation
    print("\n=== FINAL EVALUATION ===")
    final_accuracy = evaluate_count_model(model, test_loader)
    
    # Plot results
    plot_results(train_losses, train_accuracies, test_accuracies)
    
    # Save model
    model_path = os.path.join(data_dir, "count_only_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f'\nModel saved to {model_path}')
    
    print(f"\n=== SUMMARY ===")
    print(f"Best Test Accuracy: {max(test_accuracies):.2f}%")
    print(f"Final Test Accuracy: {final_accuracy:.2f}%")