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

# def count_valid_obstacles(obstacle_targets):
#     # Count how many obstacles have valid start/end
#     valid = (obstacle_targets[:, :, 0] >= 0) & (obstacle_targets[:, :, 1] >= 0)
#     return valid.sum(dim=1)  # (batch,)

# obs_count = np.array([[0, 0], [1, 2], [3, 4], [5, 6], [7, 8]])
# print("Count Valid Obstacles:", count_valid_obstacles(obs_count))


def generate_multi_obstacle_targets(labels):
    """
    Generate obstacle angle regions (min, max) for each sample in the batch.
    Wraparound regions are merged into a single (start, end) pair where end < start.
    """
    batch_targets = []
    for sample_idx in range(labels.shape[0]):
        label_row = labels[sample_idx]
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
                last_start, last_end = obstacle_regions[-1]
                merged_region = (last_start, first_end)

                # Remove first and last, add merged
                obstacle_regions = obstacle_regions[1:-1]
                obstacle_regions.append(merged_region)

        # Print and save
        if sample_idx == 0:
            print(f"Sample {sample_idx}: {len(obstacle_regions)} obstacles: {obstacle_regions}")
        batch_targets.append(obstacle_regions)

    return batch_targets


# Test generate_multi_obstacle_targets
labels = torch.tensor([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]])
targets = generate_multi_obstacle_targets(labels)
print("Generated Targets:", targets)

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

spans = labels_to_min_max(labels)
print("Min indices:", spans[0])
print("Max indices:", spans[1])

def extract_min_max_from_targets(batch_targets, device):
    """
    Converts list of obstacle tuples to two tensors: min_indices and max_indices.
    Pads with -1 where obstacles are missing.
    """
    max_obs = max(len(sample) for sample in batch_targets)
    min_indices = []
    max_indices = []

    for sample in batch_targets:
        mins = [t[0] for t in sample]
        maxs = [t[1] for t in sample]

        # Pad with -1 if fewer than max_obs
        mins += [-1] * (max_obs - len(mins))
        maxs += [-1] * (max_obs - len(maxs))

        min_indices.append(mins)
        max_indices.append(maxs)

    return (
        torch.tensor(min_indices, device=device, dtype=torch.long),
        torch.tensor(max_indices, device=device, dtype=torch.long),
    )

# Example usage
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_targets = generate_multi_obstacle_targets(labels)
min_indices, max_indices = extract_min_max_from_targets(batch_targets, device)
print("Min indices tensor:", min_indices)
print("Max indices tensor:", max_indices)




