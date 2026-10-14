import numpy as np

def is_bad_label(label_array):
    """
    Define criteria for a bad label. For example:
    - All ones (all angles marked as obstacle)
    - Or any other custom criteria you want to check.
    """
    # Example: label is all ones (size 360)
    if label_array.shape == (360,) and np.all(label_array == 1):
        return True
    # Add other conditions here if needed
    return False

def clean_dataset(data_path, labels_path, save_data_path, save_labels_path):
    # Load numpy arrays
    data = np.load(data_path)      # e.g. shape (N, 360) or (N, ...)
    labels = np.load(labels_path)  # e.g. shape (N, 360)

    if data.shape[0] != labels.shape[0]:
        raise ValueError("Data and labels must have the same number of samples")

    print(f"Original dataset size: {data.shape[0]} samples")

    # Find indices of bad labels
    bad_indices = [i for i, lbl in enumerate(labels) if is_bad_label(lbl)]

    print(f"Found {len(bad_indices)} bad labels. Removing...")

    # Keep only good indices
    good_indices = [i for i in range(len(labels)) if i not in bad_indices]

    cleaned_data = data[good_indices]
    cleaned_labels = labels[good_indices]

    print(f"Cleaned dataset size: {cleaned_data.shape[0]} samples")

    # Save cleaned arrays
    np.save(save_data_path, cleaned_data)
    np.save(save_labels_path, cleaned_labels)

    print(f"Saved cleaned data to '{save_data_path}' and labels to '{save_labels_path}'")

def view_data(data_path, labels_path):
    """
    Load and print the cleaned data and labels for verification.
    """
    data = np.load(data_path)
    labels = np.load(labels_path)

    print(f"Data Shape: {data.shape}")
    print(f"Labels Shape: {labels.shape}")

    # Print first few samples
    for i in range(min(10, len(data))):
        print(f"Sample {i} - Data: {data[i]}, Labels: {labels[i]}")

# Example usage:
# clean_dataset('lidar_data.npy', 'label_data.npy', 'lidar_data_clean.npy', 'label_data_clean.npy')
clean_dataset('pseudoslam/envs/data/new_testing_lidar.npy',
              'pseudoslam/envs/data/new_testing_labels.npy',
              'pseudoslam/envs/data/cleaned_lidar.npy',
              'pseudoslam/envs/data/cleaned_labels.npy')

view_data('pseudoslam/envs/data/cleaned_lidar.npy',
          'pseudoslam/envs/data/cleaned_labels.npy')