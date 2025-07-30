import os
import numpy as np

DATA_DIR = 'pseudoslam/envs/data'
labels_path = os.path.join(DATA_DIR, 'fix_training_labels_3obs.npy')
lidar_path = os.path.join(DATA_DIR, 'fix_training_lidar_3obs.npy')

# Load the data
fix_training_lidar_3obs = np.load(lidar_path)
fix_training_labels_3obs = np.load(labels_path)
# fix_testing_lidar_3obs = np.load(lidar_path)
# fix_testing_labels_3obs = np.load(labels_path)

# Print some of the data from fix_training_lidar_3obs.npy and fix_training_labels_3obs.npy
print(f"Shape of training lidar with 3 obstacles: {fix_training_lidar_3obs.shape}")
print(f"Shape of training labels with 3 obstacles: {fix_training_labels_3obs.shape}")
# print(f"Shape of testing lidar with 3 obstacles: {fix_testing_lidar_3obs.shape}")
# print(f"Shape of testing labels with 3 obstacles: {fix_testing_labels_3obs.shape}")

# Set to print in full
np.set_printoptions(threshold=np.inf, linewidth=200)

# Print every 10th element of fix_training_lidar_3obs.npy and fix_training_labels_3obs.npy
print("Training Lidar (every 10th element):", fix_training_lidar_3obs[::10])
print("Training Labels (every 10th element):", fix_training_labels_3obs[::10])
# print("Testing Lidar (every 10th element):", fix_testing_lidar_3obs[::10])
# print("Testing Labels (every 10th element):", fix_testing_labels_3obs[::10])

