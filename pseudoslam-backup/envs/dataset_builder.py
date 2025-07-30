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

# Mark indices of datasets with 2 obstacles
def mark_two_obstacles_indices(labels):
    if isinstance(labels, np.ndarray):
        labels = torch.tensor(labels)

    if labels.ndim == 1:
        labels = labels.unsqueeze(0)  # (360,) → (1, 360)

    counts = get_obstacle_counts(labels)
    three_obstacles_indices = torch.where(counts == 2)[0]

    return three_obstacles_indices.numpy()  # Convert to numpy array for consistency

# Extract training and testing indices with 2 obstacles
train_indices = mark_two_obstacles_indices(fix_training_labels)
test_indices = mark_two_obstacles_indices(fix_testing_labels)

# Create new lidar and labels datasets with only 2 obstacles
fix_training_lidar_2obs = fix_training_lidar[train_indices]
fix_training_labels_2obs = fix_training_labels[train_indices]
fix_testing_lidar_2obs = fix_testing_lidar[test_indices]
fix_testing_labels_2obs = fix_testing_labels[test_indices]

# Print shape and count of new datasets
print(f"Training Lidar with 2 obstacles shape: {fix_training_lidar_2obs.shape}")
print(f"Training Labels with 2 obstacles shape: {fix_training_labels_2obs.shape}")
print(f"Testing Lidar with 2 obstacles shape: {fix_testing_lidar_2obs.shape}")
print(f"Testing Labels with 2 obstacles shape: {fix_testing_labels_2obs.shape}")

# Print first dataset in full
print("First training lidar with 2 obstacles:", fix_training_lidar_2obs[0])
print("First training labels with 2 obstacles:", fix_training_labels_2obs[0])

# Save the new datasets
np.save(os.path.join("pseudoslam", "envs", "data", "fix_training_lidar_2obs.npy"), fix_training_lidar_2obs)
np.save(os.path.join("pseudoslam", "envs", "data", "fix_training_labels_2obs.npy"), fix_training_labels_2obs)
np.save(os.path.join("pseudoslam", "envs", "data", "fix_testing_lidar_2obs.npy"), fix_testing_lidar_2obs)
np.save(os.path.join("pseudoslam", "envs", "data", "fix_testing_labels_2obs.npy"), fix_testing_labels_2obs)




