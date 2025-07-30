import numpy as np
import os

DATA_DIR = 'pseudoslam/envs/data'
labels_path = os.path.join(DATA_DIR, 'fix_training_lidar_3obs.npy')
lidar_path = os.path.join(DATA_DIR, 'fix_testing_lidar_3obs.npy')

# if not (os.path.exists(labels_path) and os.path.exists(lidar_path)):
# 	raise FileNotFoundError(f"Required data files not found in '{DATA_DIR}' directory.")

# labels = np.load(labels_path)
# data = np.load(lidar_path)

# # Delete data after 23
# trimmed_labels = labels[0, :23]
# trimmed_data = data[0, :23]

# np.save(labels_path, trimmed_labels)
# np.save(lidar_path, trimmed_data)

# print(f'Shape of labels: {trimmed_labels.shape}')
# print(f'Shape of lidar: {trimmed_data.shape}')

print(f'Shape of labels: {labels.shape}')
print(f'Shape of lidar: {data.shape}')