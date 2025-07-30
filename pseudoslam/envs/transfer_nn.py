import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pseudoslam.envs.resnet_3 import lidar_to_tsdf_obstacles, generate_multi_obstacle_targets, LidarTSDFDataset, extract_min_max_from_targets, create_biased_mask, ImprovedResNetNoGating3
from pseudoslam.envs.resnet_1obs import ImprovedResNetNoGating1
from pseudoslam.envs.resnet_2obs import ImprovedResNetNoGating2
import matplotlib.pyplot as plt

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

num_epochs = 100
batch_size = 64
learning_rate = 0.0001
torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(True)
torch.cuda.empty_cache()

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

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)

class GatingModule(nn.Module):
    """
    Simplified gating module that focuses on what's actually used in the forward pass
    """
    def __init__(self, feature_dim=128, hidden_dim=64, max_obstacles=3):
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
        self.count_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 4)  # 4 classes: 0, 1, 2, 3 obstacles
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
        
        # Confidence network for each obstacle (used in loss and post-processing)
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
        
        # Predict obstacle count distribution
        global_pooled = self.global_context(features)  # (batch, hidden_dim, 1)
        global_pooled = torch.mean(global_pooled, dim=-1)  # Global average pooling: (batch, hidden_dim)
        count_logits = self.count_head(global_pooled)  # [B, 4]

        # Generate gates for each obstacle
        obstacle_gates = {}
        for i in range(self.max_obstacles):
            # Presence gate
            presence_gate = self.presence_gates[i](combined_ctx)  # (batch, 1, 360)
            
            # Confidence
            confidence = self.confidence_networks[i](combined_ctx)  # (batch, 1, 360)
            
            obstacle_gates[f'obstacle_{i}'] = {
                'presence_gate': presence_gate,
                'confidence': confidence
            }
        
        return obstacle_gates, count_logits


class TransferLearningModel(nn.Module):
    def __init__(self, pretrained_models, gating_module, feature_extractor=None):
        super().__init__()
        self.max_obstacles = 3  # Add this attribute

        # Store pretrained models
        self.resnet_1obs = pretrained_models['1_obstacle']
        self.resnet_2obs = pretrained_models['2_obstacles']
        self.resnet_3obs = pretrained_models['3_obstacles']

        # Freeze pretrained models initially
        for model in [self.resnet_1obs, self.resnet_2obs, self.resnet_3obs]:
            for param in model.parameters():
                param.requires_grad = False

        self.gating_module = gating_module
        self.feature_extractor = feature_extractor

        # TSDF processing pipeline
        self.tsdf_conv2d = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, None))
        )

        # Fusion layer for combining LiDAR and TSDF features
        self.fusion_layer = nn.Sequential(
            nn.Conv1d(129, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.combination_weights = nn.Parameter(torch.ones(3) / 3)
        self.dropout_fused = nn.Dropout(p=0.2)

    def hard_gating_approach(self, count_probs, gates, pred_1, pred_2, pred_3):
        """Hard gating: select one model based on count prediction"""
        batch_size = count_probs.shape[0]
        seq_len = 360
        
        predicted_counts = torch.argmax(count_probs, dim=1)  # shape: [B]

        # Initialize final outputs
        min_logits_final = torch.zeros(batch_size, 3, seq_len, device=count_probs.device)
        max_logits_final = torch.zeros(batch_size, 3, seq_len, device=count_probs.device)
        confidence_final = torch.zeros(batch_size, 3, seq_len, device=count_probs.device)

        # Select correct prediction set based on predicted count
        for b in range(batch_size):
            count = predicted_counts[b].item()

            if count == 0:
                continue  # leave zeros
            elif count == 1:
                min_logits_final[b, :1, :] = pred_1['min_logits'][b, :1, :]
                max_logits_final[b, :1, :] = pred_1['max_logits'][b, :1, :]
                if 'confidence' in pred_1:
                    confidence_final[b, :1, :] = pred_1['confidence'][b, :1, :]
            elif count == 2:
                min_logits_final[b, :2, :] = pred_2['min_logits'][b, :2, :]
                max_logits_final[b, :2, :] = pred_2['max_logits'][b, :2, :]
                if 'confidence' in pred_2:
                    confidence_final[b, :2, :] = pred_2['confidence'][b, :2, :]
            elif count == 3:
                min_logits_final[b, :, :] = pred_3['min_logits'][b, :, :]
                max_logits_final[b, :, :] = pred_3['max_logits'][b, :, :]
                if 'confidence' in pred_3:
                    confidence_final[b, :, :] = pred_3['confidence'][b, :, :]

        # Apply gating - CREATE NEW TENSORS TO AVOID IN-PLACE MODIFICATIONS
        gated_min_logits = []
        gated_max_logits = []
        gated_confidence = []
        
        for i in range(3):
            if f'obstacle_{i}' in gates:
                presence_gate = gates[f'obstacle_{i}']['presence_gate'].squeeze(1)  # shape: [B, 360]
                confidence_gate = gates[f'obstacle_{i}']['confidence'].squeeze(1)   # shape: [B, 360]
                
                # Combined gating effect
                combined_gate = presence_gate * confidence_gate  # shape: [B, 360]

                # Apply gate to predictions for this obstacle slot - create new tensors
                gated_min = min_logits_final[:, i, :] * combined_gate
                gated_max = max_logits_final[:, i, :] * combined_gate
                gated_conf = confidence_final[:, i, :] * combined_gate
            else:
                # No gating applied
                gated_min = min_logits_final[:, i, :]
                gated_max = max_logits_final[:, i, :]
                gated_conf = confidence_final[:, i, :]
            
            gated_min_logits.append(gated_min)
            gated_max_logits.append(gated_max)
            gated_confidence.append(gated_conf)

        # Stack the results
        final_min_logits = torch.stack(gated_min_logits, dim=1)  # (B, 3, 360)
        final_max_logits = torch.stack(gated_max_logits, dim=1)  # (B, 3, 360)
        final_confidence = torch.stack(gated_confidence, dim=1)  # (B, 3, 360)

        return {
            'min_logits': final_min_logits,
            'max_logits': final_max_logits,
            'confidence': final_confidence
        }

    def forward(self, x, tsdf_data, return_intermediate=False):
        batch_size = x.shape[0]

        # --- LiDAR feature extraction ---
        if self.feature_extractor is not None:
            lidar_feat = self.feature_extractor(x)
        else:
            lidar_feat = x

        # --- TSDF feature extraction ---
        while tsdf_data.ndim > 4:
            tsdf_data = tsdf_data.squeeze(0)

        if tsdf_data.shape[-1] != 360:
            tsdf_data = F.interpolate(tsdf_data, size=(tsdf_data.shape[2], 360), mode='bilinear', align_corners=False)

        tsdf_feat = self.tsdf_conv2d(tsdf_data).squeeze(2)

        # --- Sanity check and interpolate if needed ---
        if tsdf_feat.shape[-1] != lidar_feat.shape[-1]:
            tsdf_feat = F.interpolate(tsdf_feat, size=lidar_feat.shape[-1], mode='linear', align_corners=False)

        # --- Fusion ---
        fused_feat = torch.cat([lidar_feat, tsdf_feat], dim=1)
        fused_feat = self.fusion_layer(fused_feat)
        fused_feat = self.dropout_fused(fused_feat)

        # --- Gating decisions ---
        obstacle_gates, obstacle_count_probs = self.gating_module(fused_feat)

        # --- Pretrained model outputs ---
        pred_1 = self.resnet_1obs(x, tsdf_data)
        pred_2 = self.resnet_2obs(x, tsdf_data)
        pred_3 = self.resnet_3obs(x, tsdf_data)

        # --- Final selection with gating ---
        final_predictions = self.hard_gating_approach(
            obstacle_count_probs, obstacle_gates, pred_1, pred_2, pred_3
        )

        if return_intermediate:
            return {
                'final_predictions': final_predictions,
                'obstacle_gates': obstacle_gates,
                'obstacle_count_probs': obstacle_count_probs,
                'raw_predictions': {'1_obs': pred_1, '2_obs': pred_2, '3_obs': pred_3}
            }

        return final_predictions

    def unfreeze_pretrained_models(self):
        for model in [self.resnet_1obs, self.resnet_2obs, self.resnet_3obs]:
            for param in model.parameters():
                param.requires_grad = True

    def freeze_pretrained_models(self):
        for model in [self.resnet_1obs, self.resnet_2obs, self.resnet_3obs]:
            for param in model.parameters():
                param.requires_grad = False


class AdaptiveGatingLoss(nn.Module):
    def __init__(self, detection_weight=1.0, count_weight=3.0, presence_weight=0.2, gate_usage_weight=0.1):
        super().__init__()
        self.detection_weight = detection_weight
        self.count_weight = count_weight
        self.presence_weight = presence_weight
        self.gate_usage_weight = gate_usage_weight

        self.detection_loss = nn.CrossEntropyLoss()
        self.count_loss = nn.CrossEntropyLoss()
        self.presence_loss = nn.BCELoss()  # Changed from BCEWithLogitsLoss since gates use Sigmoid

    def forward(self, predictions_dict, targets, obstacle_count_probs, obstacle_gates, true_counts, true_presence):
        min_logits = predictions_dict['min_logits']  # (B, max_obstacles, 360)
        max_logits = predictions_dict['max_logits']  # (B, max_obstacles, 360)

        B, max_obs, _ = min_logits.shape
        detection_loss_total = 0.0

        # --- Detection Loss ---
        for i in range(max_obs):
            min_logit_i = min_logits[:, i, :]
            max_logit_i = max_logits[:, i, :]

            min_target_i = targets['min_classes'][:, i]
            max_target_i = targets['max_classes'][:, i]

            detection_loss_total += self.detection_loss(min_logit_i, min_target_i)
            detection_loss_total += self.detection_loss(max_logit_i, max_target_i)

        det_loss = detection_loss_total / (2 * max_obs)
        
        # --- Count Loss ---
        true_counts = true_counts.squeeze(1).to(obstacle_count_probs.device).long()
        count_loss = self.count_loss(obstacle_count_probs, true_counts)

        # --- Presence Loss (ensures gating components get gradients) ---
        presence_loss = 0.0
        gate_usage_loss = 0.0
        
        for i in range(max_obs):
            if f'obstacle_{i}' in obstacle_gates:
                pred_gate = obstacle_gates[f'obstacle_{i}']['presence_gate'].squeeze(1)  # (B, 360)
                confidence_gate = obstacle_gates[f'obstacle_{i}']['confidence'].squeeze(1)  # (B, 360)
                
                # Average over sequence dimension for presence prediction
                pred_gate_scalar = pred_gate.mean(dim=1)  # (B,)
                confidence_scalar = confidence_gate.mean(dim=1)  # (B,)
                
                target_gate = true_presence[:, i].to(pred_gate_scalar.device)  # (B,)
                
                # Presence loss
                presence_loss += self.presence_loss(pred_gate_scalar, target_gate)
                
                # Gate usage loss - encourages gates to be used actively
                # This ensures ALL gating components get gradients
                gate_variance = torch.var(pred_gate, dim=1).mean()  # Encourage variation across sequence
                confidence_variance = torch.var(confidence_gate, dim=1).mean()
                
                gate_usage_loss += (1.0 / (gate_variance + 1e-6)) + (1.0 / (confidence_variance + 1e-6))

        presence_loss = presence_loss / max_obs
        gate_usage_loss = gate_usage_loss / max_obs

        # --- Total Loss ---
        total_loss = (
            self.detection_weight * det_loss +
            self.count_weight * count_loss +
            self.presence_weight * presence_loss +
            self.gate_usage_weight * gate_usage_loss
        )

        return {
            'total_loss': total_loss,
            'detection_loss': det_loss,
            'count_loss': count_loss,
            'presence_loss': presence_loss,
            'gate_usage_loss': gate_usage_loss
        }

# Example usage
def create_transfer_learning_model(pretrained_model_paths, feature_dim=64):
    """
    Helper function to create the transfer learning model
    """

    def load_pretrained_model(path, num_obstacles):
        if num_obstacles == 1:
            model = ImprovedResNetNoGating1()
        elif num_obstacles == 2:
            model = ImprovedResNetNoGating2()
        elif num_obstacles == 3:
            model = ImprovedResNetNoGating3()
        model.load_state_dict(torch.load(path))  # if saved with torch.save(model.state_dict())
        return model

    pretrained_models = {
        '1_obstacle': load_pretrained_model(pretrained_model_paths['1_obstacle'], num_obstacles=1),
        '2_obstacles': load_pretrained_model(pretrained_model_paths['2_obstacles'], num_obstacles=2),
        '3_obstacles': load_pretrained_model(pretrained_model_paths['3_obstacles'], num_obstacles=3)
    }
    
    # Create gating module (using your existing implementation)
    gating_module = GatingModule()
    
    # Create transfer learning model
    model = TransferLearningModel(
        pretrained_models=pretrained_models,
        gating_module=gating_module
    )
    
    return model


base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, "data")

pretrained_model_paths = {
    '1_obstacle': os.path.join(data_dir, "resnet_1.pth"),
    '2_obstacles': os.path.join(data_dir, "resnet_2.pth"),
    '3_obstacles': os.path.join(data_dir, "resnet_3.pth")
}

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

def generate_targets_and_counts(labels):
    """Adapt your existing target generation to include counts and presence"""
    obstacle_targets = generate_multi_obstacle_targets(labels)  # Your existing function

    # Generate count targets
    count_targets = []
    presence_targets = []

    for targets in obstacle_targets:
        num_obstacles = len(targets) if targets else 0

        # Generate presence targets (binary for each of 3 possible obstacles)
        presence = [0.0, 0.0, 0.0]
        for i in range(min(num_obstacles, 3)):
            presence[i] = 1.0
        presence_targets.append(presence)

    count_targets = get_obstacle_counts(labels)
    presence_targets = torch.tensor(presence_targets, device=device, dtype=torch.float32)

    return obstacle_targets, count_targets, presence_targets


def convert_targets(targets, max_obstacles=3):
    batch_size = len(targets)
    min_classes = torch.zeros((batch_size, max_obstacles), dtype=torch.long)
    max_classes = torch.zeros((batch_size, max_obstacles), dtype=torch.long)
    presence_mask = torch.zeros((batch_size, max_obstacles), dtype=torch.bool)
    count_classes = torch.zeros((batch_size,), dtype=torch.long)

    for b, obstacles in enumerate(targets):
        num_obs = min(len(obstacles), max_obstacles)
        count_classes[b] = num_obs
        for i in range(num_obs):
            min_idx, max_idx = obstacles[i]
            min_classes[b, i] = min_idx
            max_classes[b, i] = max_idx
            presence_mask[b, i] = True

    return {
        'min_classes': min_classes,
        'max_classes': max_classes,
        'presence_mask': presence_mask,
        'count_class': count_classes
    }


def train_model(model, train_loader):
    """
    Training function adapted for transfer learning model with comprehensive monitoring
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-3)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.7, patience=3
    )
    
    loss_fn = AdaptiveGatingLoss(detection_weight=1.0, count_weight=0.3, presence_weight=0.2, gate_usage_weight=0.1)
    
    loss_history = []
    loss_components = {
        'total_loss': [],
        'detection_loss': [], 
        'count_loss': [], 
        'presence_loss': [],
        'gate_usage_loss': []
    }
    
    # Improved gradient checking
    def check_gradients():
        total_norm = 0
        param_count = 0
        no_grad_params = []
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
            else:
                no_grad_params.append(name)
        
        # Only warn about gating module parameters without gradients
        gating_no_grad = [name for name in no_grad_params if 'gating_module' in name]
        if gating_no_grad:
            print(f"WARNING: {len(gating_no_grad)} gating parameters without gradients")
        
        return (total_norm ** 0.5) / max(param_count, 1)

    
    print("Starting transfer learning training...")
    print(f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters")
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_components = {
            'total_loss': 0.0,
            'detection_loss': 0.0, 
            'count_loss': 0.0, 
            'presence_loss': 0.0,
            'gate_usage_loss': 0.0
        }
        batch_count = 0
        
        for batch_idx, (distances, labels, _, tsdf) in enumerate(train_loader):
            distances = distances.to(device)
            tsdf = tsdf.to(device)
            
            # Generate targets
            obstacle_targets, count_targets, presence_targets = generate_targets_and_counts(labels)
            
            if not any(len(targets) > 0 for targets in obstacle_targets):
                print(f"WARNING: Batch {batch_idx} has no obstacles, skipping")
                continue

            optimizer.zero_grad()
            
            # Forward pass
            results = model(distances, tsdf, return_intermediate=True)
            predictions = results['final_predictions']
            count_probs = results['obstacle_count_probs']
            gates = results['obstacle_gates']
            
            # Convert targets
            B = len(obstacle_targets)
            max_obstacles = 3

            min_classes = torch.zeros((B, max_obstacles), dtype=torch.long, device=device)
            max_classes = torch.zeros((B, max_obstacles), dtype=torch.long, device=device)

            for b in range(B):
                obstacles = obstacle_targets[b]
                for i, (min_c, max_c) in enumerate(obstacles):
                    if i >= max_obstacles:
                        break
                    min_classes[b, i] = min_c
                    max_classes[b, i] = max_c

            formatted_targets = {
                'min_classes': min_classes,
                'max_classes': max_classes
            }

            # Compute loss
            loss_dict = loss_fn(
                predictions_dict=predictions, 
                targets=formatted_targets,
                obstacle_count_probs=count_probs,
                obstacle_gates=gates,
                true_counts=count_targets,
                true_presence=presence_targets
            )

            loss = loss_dict['total_loss']
            
            if torch.isnan(loss):
                print(f"NaN loss detected at epoch {epoch}, batch {batch_idx}")
                continue
                
            loss.backward()
            
            # Check gradients
            grad_norm = check_gradients()
            if grad_norm > 10.0:
                print(f"Large gradient norm: {grad_norm:.4f}")
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Update metrics
            epoch_loss += loss.item()
            for key in epoch_components:
                if key in loss_dict:
                    epoch_components[key] += loss_dict[key].item()
            batch_count += 1
            
            if batch_idx % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], '
                      f'Total Loss: {loss.item():.4f}')
                print(f'  Detection: {loss_dict["detection_loss"].item():.4f}, '
                      f'Count: {loss_dict["count_loss"].item():.4f}, '
                      f'Presence: {loss_dict["presence_loss"].item():.4f}, '
                      f'Gate Usage: {loss_dict["gate_usage_loss"].item():.4f}')
        
        if batch_count == 0:
            continue
            
        # Calculate averages
        avg_epoch_loss = epoch_loss / batch_count
        scheduler.step(avg_epoch_loss)
        loss_history.append(avg_epoch_loss)
        
        for key in epoch_components:
            loss_components[key].append(epoch_components[key] / batch_count)
        
        # Print epoch summary
        print(f'Epoch [{epoch+1}/{num_epochs}] Average Loss: {avg_epoch_loss:.4f}')
        print(f'  Detection Loss: {epoch_components["detection_loss"]/batch_count:.4f}')
        print(f'  Count Loss: {epoch_components["count_loss"]/batch_count:.4f}')
        print(f'  Presence Loss: {epoch_components["presence_loss"]/batch_count:.4f}')
        print(f'  Gate Usage Loss: {epoch_components["gate_usage_loss"]/batch_count:.4f}')
        
        # Phase-based training
        if epoch == 10:
            print("Phase 2: Unfreezing pretrained models for fine-tuning...")
            model.unfreeze_pretrained_models()
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1
        
        # Early stopping
        if len(loss_history) > 15:
            recent_losses = loss_history[-15:]
            if max(recent_losses) - min(recent_losses) < 0.001:
                print("Loss plateaued, stopping early")
                break
    
    return loss_history, loss_components

def evaluate_transfer_model(model, test_loader, loss_history, loss_components, device='cuda'):
    """
    Comprehensive evaluation function for the transfer learning model
    Adapted from your original test function with improved processing and visualization
    """
    
    model.eval()
    total_samples = 0
    total_obstacles = 0
    matched_obstacles = 0
    total_absolute_error = []
    tolerance = 5  # degrees
    num_batches = 0
    
    # Count prediction metrics
    correct_counts = 0
    count_confusion = np.zeros((4, 4))  # 4x4 for 0,1,2,3 obstacles
    
    # Gating analysis
    gating_stats = {
        'presence_activations': [[] for _ in range(3)],
        'confidence_scores': [[] for _ in range(3)],
        'model_selections': []  # Track which pretrained model is selected most
    }
    
    print("Starting evaluation...")
    
    with torch.no_grad():
        for batch_idx, (distances, labels, _, tsdf) in enumerate(test_loader, start=1):
            gt_obstacle_targets = generate_multi_obstacle_targets(labels)  # Your existing function
            distances = distances.to(device)
            tsdf = tsdf.to(device)
            
            # Generate count and presence targets
            obstacle_targets, count_targets, presence_targets = generate_targets_and_counts(labels)
            
            # Forward pass through transfer model
            results = model(distances, tsdf, return_intermediate=True)
            predictions = results['final_predictions']
            count_probs = results['obstacle_count_probs']
            gates = results['obstacle_gates']
            raw_predictions = results['raw_predictions']
            
            # Use your improved post-processing (you'll need to adapt this for transfer model output)
            batch_predicted_spans = improved_post_processing_transfer(results)
            
            batch_size = distances.size(0)
            total_samples += batch_size
            
            # Count prediction accuracy
            predicted_counts = torch.argmax(count_probs, dim=1)
            if len(predicted_counts.shape) > 1:  # Handle sequence dimension
                predicted_counts = predicted_counts[:, 0]  # Take first position
                count_targets_eval = count_targets
            else:
                count_targets_eval = count_targets
                
            correct_counts += (predicted_counts == count_targets_eval.to(predicted_counts.device)).sum().item()
            
            # Update confusion matrix
            for i in range(batch_size):
                true_count = int(count_targets_eval[i].item())
                pred_count = int(predicted_counts[i].item())
                count_confusion[true_count, pred_count] += 1
            
            # Collect gating statistics
            for i in range(3):
                if f'obstacle_{i}' in gates:
                    presence = gates[f'obstacle_{i}']['presence_gate'].mean(dim=-1).cpu().numpy()  # Average over sequence
                    confidence = gates[f'obstacle_{i}']['confidence'].mean(dim=-1).cpu().numpy()
                    
                    gating_stats['presence_activations'][i].extend(presence.flatten())
                    gating_stats['confidence_scores'][i].extend(confidence.flatten())
            
            # Track which model would be selected (for analysis)
            model_selections = torch.argmax(count_probs, dim=1).cpu().numpy()
            if len(model_selections.shape) > 1:
                model_selections = model_selections[:, 0]  # Take first position
            gating_stats['model_selections'].extend(model_selections)
            
            # Obstacle matching (your original logic)
            for b in range(batch_size):
                gt_spans = gt_obstacle_targets[b]
                pred_spans = batch_predicted_spans[b]
                
                # Debug output for some samples
                if num_batches <= 3 and b < 5:
                    print(f"Batch {num_batches}, Sample {b+1}:")
                    print(f"  Predicted Spans: {pred_spans}")
                    print(f"  Ground Truth Spans: {gt_spans}")
                    print(f"  Predicted Count: {predicted_counts[b].item()}, True Count: {count_targets_eval[b].item()}")
                    
                    # Show gating information
                    for i in range(3):
                        if f'obstacle_{i}' in gates:
                            presence_val = gates[f'obstacle_{i}']['presence_gate'][b].mean().item()
                            confidence_val = gates[f'obstacle_{i}']['confidence'][b].mean().item()
                            print(f"  Gate {i}: Presence={presence_val:.3f}, Confidence={confidence_val:.3f}")
                
                total_obstacles += len(gt_spans)
                
                # Match predicted spans to ground truth
                for gt_min, gt_max in gt_spans:
                    gt_width = (gt_max - gt_min + 360) % 360 + 1
                    best_error = float('inf')
                    
                    for pred_min, pred_max in pred_spans:
                        pred_width = (pred_max - pred_min + 360) % 360 + 1
                        error = abs(pred_width - gt_width)
                        best_error = min(best_error, error)
                    
                    if best_error != float('inf'):
                        total_absolute_error.append(best_error)
                        if best_error <= tolerance:
                            matched_obstacles += 1
            
            num_batches += 1
    
    # Calculate metrics
    count_accuracy = correct_counts / total_samples
    avg_error = np.mean(total_absolute_error) if total_absolute_error else float('inf')
    obstacle_accuracy = 100.0 * matched_obstacles / total_obstacles if total_obstacles > 0 else 0
    
    print(f'\n=== EVALUATION RESULTS ===')
    print(f'Total Samples: {total_samples}')
    print(f'Count Prediction Accuracy: {count_accuracy:.4f}')
    print(f'Total GT Obstacles: {total_obstacles}')
    print(f'Matched Obstacles (±{tolerance}° width): {matched_obstacles}')
    print(f'Obstacle Detection Accuracy: {obstacle_accuracy:.2f}%')
    print(f'Average Width Error: {avg_error:.2f} degrees')
    
    # Print count confusion matrix
    print(f'\nCount Prediction Confusion Matrix:')
    print(f'True\\Pred\t0\t1\t2\t3')
    for i in range(4):
        row_str = f'{i}\t\t'
        for j in range(4):
            row_str += f'{int(count_confusion[i,j])}\t'
        print(row_str)
    
    # Create comprehensive visualization
    plt.figure(figsize=(20, 15))
    num_epochs = len(loss_history)
    
    # 1. Training Loss
    plt.subplot(3, 4, 1)
    plt.plot(range(num_epochs), loss_history, label='Total Loss', linewidth=2)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Loss Components
    plt.subplot(3, 4, 2)
    for key, values in loss_components.items():
        if values is None or len(values) == 0:
            continue
        
        # Convert tensors to CPU + numpy
        if hasattr(values, 'cpu'):
            values = values.cpu().numpy()
        elif isinstance(values, list):
            values = [v.cpu().item() if hasattr(v, 'cpu') else (v.item() if hasattr(v, 'item') else v) for v in values]
        
        # Skip if wrong shape
        if len(values) != num_epochs:
            print(f"Skipping '{key}' due to mismatched length: expected {num_epochs}, got {len(values)}")
            continue
            
        plt.plot(range(num_epochs), values, label=key, linewidth=2)
    
    plt.xlabel('Epochs')
    plt.ylabel('Loss Component')
    plt.title('Loss Components')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Count Prediction Confusion Matrix
    plt.subplot(3, 4, 3)
    im = plt.imshow(count_confusion, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Count Prediction\nConfusion Matrix')
    plt.colorbar(im)
    tick_marks = np.arange(4)
    plt.xticks(tick_marks, ['0', '1', '2', '3'])
    plt.yticks(tick_marks, ['0', '1', '2', '3'])
    plt.xlabel('Predicted Count')
    plt.ylabel('True Count')
    
    # Add text annotations
    for i in range(4):
        for j in range(4):
            plt.text(j, i, f'{int(count_confusion[i, j])}', 
                    horizontalalignment="center", color="white" if count_confusion[i, j] > count_confusion.max()/2 else "black")
    
    # 4. Model Selection Distribution
    plt.subplot(3, 4, 4)
    model_counts = np.bincount(gating_stats['model_selections'], minlength=4)
    plt.bar(['0 obs', '1 obs', '2 obs', '3 obs'], model_counts)
    plt.title('Model Selection\nDistribution')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
    
    # 5-7. Presence Gate Activations
    for i in range(3):
        plt.subplot(3, 4, 5+i)
        if gating_stats['presence_activations'][i]:
            plt.hist(gating_stats['presence_activations'][i], bins=30, alpha=0.7, density=True)
            plt.title(f'Obstacle {i+1}\nPresence Gate')
            plt.xlabel('Activation')
            plt.ylabel('Density')
            plt.axvline(x=0.5, color='r', linestyle='--', alpha=0.7, label='Threshold')
            plt.legend()
        else:
            plt.text(0.5, 0.5, 'No Data', ha='center', va='center')
            plt.title(f'Obstacle {i+1}\nPresence Gate')
    
    # 8-10. Confidence Score Distributions
    for i in range(3):
        plt.subplot(3, 4, 8+i)
        if gating_stats['confidence_scores'][i]:
            plt.hist(gating_stats['confidence_scores'][i], bins=30, alpha=0.7, density=True, color='orange')
            plt.title(f'Obstacle {i+1}\nConfidence Score')
            plt.xlabel('Confidence')
            plt.ylabel('Density')
        else:
            plt.text(0.5, 0.5, 'No Data', ha='center', va='center')
            plt.title(f'Obstacle {i+1}\nConfidence Score')
    
    # 11. Detection Accuracy Summary
    plt.subplot(3, 4, 11)
    metrics = ['Count Acc', 'Detection Acc', 'Avg Error']
    values = [count_accuracy * 100, obstacle_accuracy, avg_error]
    colors = ['green' if v > 70 else 'orange' if v > 50 else 'red' for v in values[:2]] + ['red' if values[2] > 10 else 'orange' if values[2] > 5 else 'green']
    
    bars = plt.bar(metrics, values, color=colors, alpha=0.7)
    plt.title('Performance Summary')
    plt.ylabel('Accuracy (%) / Error (°)')
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{value:.1f}', ha='center', va='bottom')
    
    # 12. Gating Statistics Summary
    plt.subplot(3, 4, 12)
    gate_means = [np.mean(gating_stats['presence_activations'][i]) if gating_stats['presence_activations'][i] else 0 
                  for i in range(3)]
    conf_means = [np.mean(gating_stats['confidence_scores'][i]) if gating_stats['confidence_scores'][i] else 0 
                  for i in range(3)]
    
    x = np.arange(3)
    width = 0.35
    
    plt.bar(x - width/2, gate_means, width, label='Presence', alpha=0.7)
    plt.bar(x + width/2, conf_means, width, label='Confidence', alpha=0.7)
    
    plt.xlabel('Obstacle Index')
    plt.ylabel('Average Score')
    plt.title('Average Gating Scores')
    plt.xticks(x, ['Obs 1', 'Obs 2', 'Obs 3'])
    plt.legend()
    plt.ylim(0, 1)
    
    plt.tight_layout()
    plt.show()
    
    return {
        'count_accuracy': count_accuracy,
        'obstacle_accuracy': obstacle_accuracy,
        'avg_error': avg_error,
        'count_confusion': count_confusion,
        'gating_stats': gating_stats
    }


def improved_post_processing_transfer(results):
    """
    Post-process transfer learning model output to extract obstacle predictions.
    Assumes `results` is a dictionary with:
        - 'obstacle_gates': dict of presence/confidence gates per slot
        - 'obstacle_count_probs': (B, 4)
        - 'final_predictions': dict with 'min_logits' and 'max_logits', each of shape (B, 3, 360)
    Returns:
        List of obstacle spans (min_idx, max_idx) per batch item.
    """
    gates = results['obstacle_gates']  # Dict[str -> {'presence_gate': (B, 1, 360), 'confidence': (B, 1, 360)}]
    count_probs = results['obstacle_count_probs']  # (B, 4)
    min_logits_all = results['final_predictions']['min_logits']  # (B, 3, 360)
    max_logits_all = results['final_predictions']['max_logits']  # (B, 3, 360)

    batch_size = count_probs.shape[0]
    batch_results = []

    for b in range(batch_size):
        count_pred = torch.argmax(count_probs[b]).item()  # scalar in [0, 3]

        min_logits = min_logits_all[b]  # (3, 360)
        max_logits = max_logits_all[b]  # (3, 360)

        min_probs = torch.softmax(min_logits, dim=-1).detach().cpu().numpy()  # (3, 360)
        max_probs = torch.softmax(max_logits, dim=-1).detach().cpu().numpy()

        obstacle_candidates = []

        for slot_idx in range(3):
            # Gate check (if available)
            gate_key = f'obstacle_{slot_idx}'
            if gate_key in gates:
                presence_gate_val = gates[gate_key]['presence_gate'][b].mean().item()
                confidence_gate_val = gates[gate_key]['confidence'][b].mean().item()

                if presence_gate_val < 0.3 or confidence_gate_val < 0.2:
                    continue
            else:
                continue  # Skip if gate info missing

            # Argmax for span
            best_min_idx = np.argmax(min_probs[slot_idx])
            best_max_idx = np.argmax(max_probs[slot_idx])

            if best_min_idx == best_max_idx:
                best_min_idx = (best_min_idx - 3) % 360
                best_max_idx = (best_max_idx + 3) % 360

            # Handle wraparound logic
            if best_min_idx <= best_max_idx:
                span_width = best_max_idx - best_min_idx + 1
            else:
                span_width = (360 - best_min_idx) + (best_max_idx + 1)

            if span_width < 1 or span_width > 200:
                continue

            min_score = min_probs[slot_idx, best_min_idx]
            max_score = max_probs[slot_idx, best_max_idx]
            combined_score = (min_score + max_score) / 2.0

            # Boost confidence
            combined_score *= (1.0 + confidence_gate_val)

            obstacle_candidates.append({
                'min_idx': best_min_idx,
                'max_idx': best_max_idx,
                'score': combined_score,
                'span_width': span_width,
                'slot': slot_idx
            })

        # Sort candidates by descending score
        obstacle_candidates.sort(key=lambda x: x['score'], reverse=True)

        # Select top-k non-overlapping spans
        selected_obstacles = []
        used_indices = set()

        for candidate in obstacle_candidates:
            if candidate['min_idx'] <= candidate['max_idx']:
                span_indices = set(range(candidate['min_idx'], candidate['max_idx'] + 1))
            else:
                span_indices = set(range(candidate['min_idx'], 360)) | set(range(0, candidate['max_idx'] + 1))

            if len(span_indices & used_indices) == 0:
                selected_obstacles.append((candidate['min_idx'], candidate['max_idx']))
                used_indices.update(span_indices)

                if len(selected_obstacles) >= count_pred:
                    break

        # Pad with dummy spans if needed
        while len(selected_obstacles) < count_pred:
            selected_obstacles.append((0, 10))

        batch_results.append(selected_obstacles)

    return batch_results



def convert_transfer_predictions_to_standard_format(results):
    """
    Helper function to convert transfer learning results to your standard prediction format
    for compatibility with existing functions
    """
    predictions_dict = {
        'min_logits': results['final_predictions'][:, :, 0, :],  # (B, max_obstacles, 360)
        'max_logits': results['final_predictions'][:, :, 1, :],  # (B, max_obstacles, 360)
        'count_pred': torch.argmax(results['obstacle_count_probs'], dim=1)  # (B,)
    }
    
    # Handle sequence dimension in count predictions if present
    if len(results['obstacle_count_probs'].shape) == 3:
        predictions_dict['count_pred'] = torch.argmax(results['obstacle_count_probs'], dim=1)[:, 0]
    
    return predictions_dict

if __name__ == "__main__":
    model = create_transfer_learning_model(pretrained_model_paths, feature_dim=64)
    model.to(device)
    loss_history, loss_components = train_model(model, train_loader)
    evaluate_transfer_model(model, test_loader, loss_history, loss_components, device=device)
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

    model_path = os.path.join(data_dir, "transfer_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f'Model saved to {model_path}')