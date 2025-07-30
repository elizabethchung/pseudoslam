import numpy as np
import torch

def labels_to_min_max(labels):
    """
    For each label row, find all contiguous ranges of 1s (with wraparound).
    Returns two lists:
      - min_indices: list of lists of min indices for each range in each row
      - max_indices: list of lists of max indices for each range in each row
    """
    labels = np.atleast_2d(labels)
    min_indices_all, max_indices_all = [], []
    for row in labels:
        row = np.asarray(row)
        n = row.shape[0]
        # Find all indices where value is 1
        ones = np.where(row == 1)[0]
        if len(ones) == 0:
            min_indices_all.append([0])
            max_indices_all.append([n - 1])
            continue
        # Find breaks between contiguous regions
        diffs = np.diff(np.concatenate(([ones[-1] - n], ones)))
        breaks = np.where(diffs != 1)[0]
        # Split into contiguous regions
        ranges = np.split(ones, breaks[1:])
        min_indices, max_indices = [], []
        for r in ranges:
            if len(r) == 0:
                continue
            min_indices.append(r[0])
            max_indices.append(r[-1])
        # Handle wraparound: if first and last region are contiguous
        if len(ranges) > 1 and (ranges[0][0] == 0 and ranges[-1][-1] == n - 1):
            # Merge first and last
            merged = np.concatenate((ranges[-1], ranges[0]))
            min_indices = [merged[0]] + min_indices[1:-1]
            max_indices = max_indices[1:-1] + [merged[-1]]
        min_indices_all.append(min_indices)
        max_indices_all.append(max_indices)
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


sample_labels = np.array([
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
])

indices = labels_to_min_max(sample_labels.reshape(1, -1))
print("Min indices:", indices[0][0])
print("Max indices:", indices[1][0])

targets = generate_multi_obstacle_targets(sample_labels.reshape(1, -1))
print("Generated targets:", targets)
