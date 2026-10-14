import os
import numpy as np

DATA_DIR = 'pseudoslam/envs/data'
labels_path = os.path.join(DATA_DIR, 'training_labels.npy')
lidar_path = os.path.join(DATA_DIR, 'training_lidar.npy')

if not (os.path.exists(labels_path) and os.path.exists(lidar_path)):
	raise FileNotFoundError(f"Required data files not found in '{DATA_DIR}' directory.")

labels = np.load(labels_path)
data = np.load(lidar_path)

# Print 100 random samples from the lidar and labels
num_samples = 100
if data.shape[0] < num_samples or labels.shape[0] < num_samples:
    raise ValueError(f"Dataset contains fewer than {num_samples} samples.")
indices = np.random.choice(data.shape[0], num_samples, replace=False)
print(f"Randomly selected {num_samples} samples from the dataset:")
for i in indices:
    print(f"Sample {i} - Data: {data[i]}, Labels: {labels[i]}")