from torch.utils.data import Dataset, DataLoader    
import torch

class DummyTwoObstacleDataset(Dataset):
    def __init__(self, num_samples=100):
        self.num_samples = num_samples
        # Hardcoded angular positions
        self.regions = torch.tensor([[30, 50], [180, 200]])  # shape (2, 2)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        distances = torch.zeros(1, 360)  # shape: (1, 360)
        tsdf = torch.zeros(1, 360)
        label = self.regions.clone()     # shape: (2, 2) → [[min1, max1], [min2, max2]]
        return distances, label, 2, tsdf
    
dummy_loader = DataLoader(DummyTwoObstacleDataset(), batch_size=16, shuffle=True)



resnet.train()
for epoch in range(10):
    total_loss = 0
    for distances, labels, _, tsdf in dummy_loader:
        distances = distances.to(device)
        tsdf = tsdf.to(device)
        labels = labels.to(device)

        min_logits, max_logits, _ = resnet(distances, tsdf)  # [B, 2, 360]
        
        ce_loss = 0.0
        for i in range(2):  # 2 obstacles
            min_target = labels[:, i, 0]
            max_target = labels[:, i, 1]

            ce_min = F.cross_entropy(min_logits[:, i, :], min_target)
            ce_max = F.cross_entropy(max_logits[:, i, :], max_target)
            ce_loss += ce_min + ce_max

        ce_loss = ce_loss / 2.0  # average over obstacles

        optimizer.zero_grad()
        ce_loss.backward()
        optimizer.step()

        total_loss += ce_loss.item()

    print(f"Epoch {epoch+1}, Avg Loss: {total_loss / len(dummy_loader):.4f}")

