import torch
import torch.nn.functional as F



def improved_multi_obstacle_loss_with_constraints(predictions, obstacle_targets,
                                                alpha=0.5,
                                                confidence_weight=1.0,
                                                span_constraint_weight=0.3,
                                                diversity_weight=1.0):
    """
    Enhanced loss function with span constraints and diversity enforcement
    """
    
    min_pred_logits = predictions['min_logits']  # (B, 3, 360)
    max_pred_logits = predictions['max_logits']  # (B, 3, 360)
    confidence_logits = predictions['confidence']  # (B, 3, 360)
    
    batch_size = min_pred_logits.shape[0]
    max_obstacles = min_pred_logits.shape[1]
    
    total_position_loss = 0.0
    total_confidence_loss = 0.0
    total_span_constraint_loss = 0.0
    total_diversity_loss = 0.0
    valid_samples = 0
    
    for batch_idx in range(batch_size):
        obstacle_regions = obstacle_targets[batch_idx]
        num_obstacles = len(obstacle_regions) if obstacle_regions else 0
        
        # Get predictions for this sample
        min_probs = torch.softmax(min_pred_logits[batch_idx], dim=-1)  # (3, 360)
        max_probs = torch.softmax(max_pred_logits[batch_idx], dim=-1)  # (3, 360)
        
        # Process each obstacle slot
        for slot_idx in range(max_obstacles):
            if slot_idx < num_obstacles:
                region = obstacle_regions[slot_idx]
                
                min_class = region[0]
                max_class = region[1]
                
                # Calculate ground truth span width
                if min_class <= max_class:
                    gt_span_width = max_class - min_class + 1
                else:
                    gt_span_width = (360 - min_class) + (max_class + 1)
                
                # Position loss
                min_target = torch.tensor(min_class, dtype=torch.long, device=min_pred_logits.device)
                max_target = torch.tensor(max_class, dtype=torch.long, device=max_pred_logits.device)
                
                ce_min = F.cross_entropy(min_pred_logits[batch_idx, slot_idx:slot_idx+1], 
                                       min_target.unsqueeze(0))
                ce_max = F.cross_entropy(max_pred_logits[batch_idx, slot_idx:slot_idx+1], 
                                       max_target.unsqueeze(0))
                
                total_position_loss += ce_min + ce_max
                
                # Confidence targets
                confidence_target = torch.zeros_like(confidence_logits[batch_idx, slot_idx])
                
                if min_class <= max_class:
                    confidence_target[min_class:max_class + 1] = 1.0
                else:
                    confidence_target[min_class:] = 1.0
                    confidence_target[:max_class + 1] = 1.0
                
                conf_loss = F.binary_cross_entropy_with_logits(
                    confidence_logits[batch_idx, slot_idx], 
                    confidence_target
                )
                total_confidence_loss += conf_loss
                
                # SPAN CONSTRAINT LOSS: Penalize unrealistic span widths
                pred_min_idx = torch.argmax(min_pred_logits[batch_idx, slot_idx])
                pred_max_idx = torch.argmax(max_pred_logits[batch_idx, slot_idx])
                
                # Calculate predicted span width
                if pred_min_idx <= pred_max_idx:
                    pred_span_width = pred_max_idx - pred_min_idx + 1
                else:
                    pred_span_width = (360 - pred_min_idx) + (pred_max_idx + 1)
                
                # Penalize spans that are too wide (>200 degrees)
                span_width_penalty = 0.0
                if pred_span_width > 200:
                    span_width_penalty += (pred_span_width - 200) ** 2 * 0.01

                # Penalize spans that are very different from ground truth width
                width_diff = abs(pred_span_width - gt_span_width)
                if width_diff > 10:  # Allow some tolerance
                    span_width_penalty += (width_diff - 10) ** 2 * 0.05

                total_span_constraint_loss += span_width_penalty
                
                valid_samples += 1
            else:
                # No obstacle - confidence should be low everywhere
                conf_loss = F.binary_cross_entropy_with_logits(
                    confidence_logits[batch_idx, slot_idx],
                    torch.zeros_like(confidence_logits[batch_idx, slot_idx])
                )
                total_confidence_loss += conf_loss
                valid_samples += 1
        
        # DIVERSITY LOSS: Encourage different predictions for different slots
        if num_obstacles >= 3:
            for i in range(max_obstacles):
                for j in range(i + 1, max_obstacles):
                    # Penalize if min predictions are too similar
                    min_similarity = torch.sum(min_probs[i] * min_probs[j])
                    max_similarity = torch.sum(max_probs[i] * max_probs[j])
                    
                    # We want low similarity (more diversity)
                    diversity_penalty = (min_similarity + max_similarity) * 0.1
                    total_diversity_loss += diversity_penalty
    
    # Normalize losses
    if valid_samples > 0:
        avg_position_loss = total_position_loss / valid_samples
        avg_confidence_loss = total_confidence_loss / valid_samples
        avg_span_constraint_loss = total_span_constraint_loss / valid_samples
        avg_diversity_loss = total_diversity_loss / max(batch_size, 1)
    else:
        avg_position_loss = torch.tensor(0.0, device=min_pred_logits.device)
        avg_confidence_loss = torch.tensor(0.0, device=min_pred_logits.device)
        avg_span_constraint_loss = torch.tensor(0.0, device=min_pred_logits.device)
        avg_diversity_loss = torch.tensor(0.0, device=min_pred_logits.device)
    
    total_loss = (alpha * avg_position_loss + 
                  confidence_weight * avg_confidence_loss +
                  span_constraint_weight * avg_span_constraint_loss +
                  diversity_weight * avg_diversity_loss)
    
    return total_loss, {
        'position_loss': avg_position_loss.item() if hasattr(avg_position_loss, "item") else avg_position_loss,
        'confidence_loss': avg_confidence_loss.item() if hasattr(avg_confidence_loss, "item") else avg_confidence_loss,
        'span_constraint_loss': avg_span_constraint_loss.item() if hasattr(avg_span_constraint_loss, "item") else avg_span_constraint_loss,
        'diversity_loss': avg_diversity_loss.item() if hasattr(avg_diversity_loss, "item") else avg_diversity_loss
    }


batch_size = 2
max_obstacles = 3
num_classes = 360  # angular classes

# Ground truth obstacle regions (min_idx, max_idx) per sample
obstacle_targets = [
    [(50, 70), (100, 120), (200, 220)],    # sample 1 with 3 obstacles
    [(10, 30), (60, 90), (150, 180)]       # sample 2 with 3 obstacles
]

# Create logits with sharp peaks at correct classes for min and max
def create_perfect_logits(obstacle_targets, batch_size, max_obstacles, num_classes):
    min_logits = torch.full((batch_size, max_obstacles, num_classes), -10.0)
    max_logits = torch.full((batch_size, max_obstacles, num_classes), -10.0)
    for b in range(batch_size):
        for i, (min_idx, max_idx) in enumerate(obstacle_targets[b]):
            min_logits[b, i, min_idx] = 10.0  # strong positive logit at GT min
            max_logits[b, i, max_idx] = 10.0  # strong positive logit at GT max
    return min_logits, max_logits

min_logits, max_logits = create_perfect_logits(obstacle_targets, batch_size, max_obstacles, num_classes)

# Create confidence logits: high inside GT span, very low outside
confidence_logits = torch.full((batch_size, max_obstacles, num_classes), -20.0)
for b in range(batch_size):
    for i, (min_idx, max_idx) in enumerate(obstacle_targets[b]):
        if min_idx <= max_idx:
            confidence_logits[b, i, min_idx:max_idx + 1] = 20.0  # sigmoid near 1 inside span
        else:
            confidence_logits[b, i, min_idx:] = 20.0
            confidence_logits[b, i, :max_idx + 1] = 20.0

predictions = {
    'min_logits': min_logits,
    'max_logits': max_logits,
    'confidence': confidence_logits
}

# Call your loss function
loss, loss_components = improved_multi_obstacle_loss_with_constraints(predictions, obstacle_targets)

print(f"Total Loss: {loss.item()}")
print("Loss Components:", loss_components)
