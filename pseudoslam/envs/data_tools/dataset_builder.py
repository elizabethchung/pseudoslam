import os
import numpy as np
import torch


# Load training data
fix_training_lidar = np.load(os.path.join("pseudoslam", "envs", "data", "fix_training_lidar.npy"))
fix_training_labels = np.load(os.path.join("pseudoslam", "envs", "data", "fix_training_labels.npy"))

# Load testing data
fix_testing_lidar = np.load(os.path.join("pseudoslam", "envs", "data", "fix_testing_lidar.npy"))
fix_testing_labels = np.load(os.path.join("pseudoslam", "envs", "data", "fix_testing_labels.npy"))


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

# Mark indices of datasets with 1 obstacle
def mark_one_obstacles_indices(labels):
    if isinstance(labels, np.ndarray):
        labels = torch.tensor(labels)

    if labels.ndim == 1:
        labels = labels.unsqueeze(0)  # (360,) → (1, 360)

    counts = get_obstacle_counts(labels)
    one_obstacles_indices = torch.where(counts == 1)[0]

    return one_obstacles_indices.numpy()  # Convert to numpy array for consistency

# Extract training and testing indices with 1 obstacle
train_indices = mark_one_obstacles_indices(fix_training_labels)
test_indices = mark_one_obstacles_indices(fix_testing_labels)

# Create new lidar and labels datasets with only 1 obstacle
fix_training_lidar_1obs = fix_training_lidar[train_indices]
fix_training_labels_1obs = fix_training_labels[train_indices]
fix_testing_lidar_1obs = fix_testing_lidar[test_indices]
fix_testing_labels_1obs = fix_testing_labels[test_indices]
fix_validation_lidar_1obs = fix_training_lidar_1obs[2500:3000]  # Use 500 samples for validation
fix_validation_labels_1obs = fix_training_labels_1obs[2500:3000]  # Use 500 samples for validation

# Crop training to 2500 samples
if fix_training_lidar_1obs.shape[0] > 2500:
    fix_training_lidar_1obs = fix_training_lidar_1obs[:2500]
    fix_training_labels_1obs = fix_training_labels_1obs[:2500]

# Print shape and count of new datasets
print(f"Training Lidar with 1 obstacle shape: {fix_training_lidar_1obs.shape}")
print(f"Training Labels with 1 obstacle shape: {fix_training_labels_1obs.shape}")
print(f"Testing Lidar with 1 obstacle shape: {fix_testing_lidar_1obs.shape}")
print(f"Testing Labels with 1 obstacle shape: {fix_testing_labels_1obs.shape}")
print(f"Validation Lidar with 1 obstacle shape: {fix_validation_lidar_1obs.shape}")
print(f"Validation Labels with 1 obstacle shape: {fix_validation_labels_1obs.shape}")

# Print first dataset in full
print("First training lidar with 1 obstacle:", fix_training_lidar_1obs[0])
print("First training labels with 1 obstacle:", fix_training_labels_1obs[0])

# Save the new datasets
np.save(os.path.join("pseudoslam", "envs", "data", "fix_training_lidar_1obs.npy"), fix_training_lidar_1obs)
np.save(os.path.join("pseudoslam", "envs", "data", "fix_training_labels_1obs.npy"), fix_training_labels_1obs)
np.save(os.path.join("pseudoslam", "envs", "data", "fix_testing_lidar_1obs.npy"), fix_testing_lidar_1obs)
np.save(os.path.join("pseudoslam", "envs", "data", "fix_testing_labels_1obs.npy"), fix_testing_labels_1obs)
np.save(os.path.join("pseudoslam", "envs", "data", "fix_validation_lidar_1obs.npy"), fix_validation_lidar_1obs)
np.save(os.path.join("pseudoslam", "envs", "data", "fix_validation_labels_1obs.npy"), fix_validation_labels_1obs)
print("New datasets saved successfully.")




