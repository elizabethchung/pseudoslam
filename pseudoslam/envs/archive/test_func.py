import numpy as np
import torch

def labels_to_min_max(labels):
    labels = np.atleast_2d(labels)
    min_indices, max_indices = [], []
    for row in labels:
        ones_indices = np.where(row == 1)[0]
        if len(ones_indices) == 0:
            min_indices.append(0)
            max_indices.append(row.shape[0] - 1)
        elif len(ones_indices) == 1:
            min_indices.append(ones_indices[0])
            max_indices.append(ones_indices[0])
        else:
            sorted_indices = np.sort(ones_indices)
            gaps = np.diff(sorted_indices)
            wrap_gap = (row.shape[0] - sorted_indices[-1]) + sorted_indices[0]
            max_gap_idx = np.argmax(gaps)
            max_gap = gaps[max_gap_idx] if gaps.size > 0 else -1
            if wrap_gap > max_gap:
                min_indices.append(sorted_indices[0])
                max_indices.append(sorted_indices[-1])
            else:
                min_indices.append(sorted_indices[max_gap_idx + 1])
                max_indices.append(sorted_indices[max_gap_idx])
    return np.array(min_indices), np.array(max_indices)

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

data = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

# min_indices, max_indices = labels_to_min_max(data)
# print("Min Indices:", min_indices)
# print("Max Indices:", max_indices)
counts = get_obstacle_counts(data)
print("Obstacle Counts:", counts)
