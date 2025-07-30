import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_model_with_gating_loader(
    model, test_loader, generate_multi_obstacle_targets,
    loss_history, loss_components, num_epochs,
    device=device, peak_threshold=0.7, tolerance=10
):
    model.eval()
    total_samples = 0
    total_obstacles = 0
    matched_obstacles = 0
    total_absolute_error = []

    gate_stats = {
        'min_gate_avg': 0.0,
        'max_gate_avg': 0.0,
        'uncertainty_avg': 0.0,
        'confidence_avg': 0.0,
    }
    num_batches = 0

    for batch_idx, (distances, labels, _, tsdf) in enumerate(test_loader, start=1):
        gt_obstacle_targets = generate_multi_obstacle_targets(labels)

        distances = distances.to(device)
        tsdf = tsdf.to(device)

        # Debug print shapes and devices before model call
        print(f"Batch {batch_idx}:")
        print(f"  distances shape: {distances.shape}, device: {distances.device}")
        print(f"  tsdf shape: {tsdf.shape}, device: {tsdf.device}")
        print(f"  model device: {next(model.parameters()).device}")
        # Print tsdf
        print(f"  tsdf: {tsdf}")

        # Forward pass
        obstacle_outputs = model(distances, tsdf)

        min_pred_logits = obstacle_outputs[0]['min_logits']
        max_pred_logits = obstacle_outputs[0]['max_logits']
        gating_info = {
            'min_gate': obstacle_outputs[0]['min_gate'],
            'max_gate': obstacle_outputs[0]['max_gate'],
            'uncertainty': obstacle_outputs[0]['uncertainty'],
            'confidence': obstacle_outputs[0]['confidence'],
        }

        pred_min_scores = torch.sigmoid(min_pred_logits).cpu().detach().numpy()
        pred_max_scores = torch.sigmoid(max_pred_logits).cpu().detach().numpy()

        batch_size = distances.size(0)
        total_samples += batch_size

        for b in range(batch_size):
            gt_spans = gt_obstacle_targets[b]
            min_peaks, _ = find_peaks(pred_min_scores[b], height=peak_threshold)
            max_peaks, _ = find_peaks(pred_max_scores[b], height=peak_threshold)

            pred_spans = []
            for min_p in min_peaks:
                max_candidates = max_peaks[max_peaks >= min_p]
                if len(max_candidates) == 0 and len(max_peaks) > 0:
                    max_p = max_peaks[0]
                elif len(max_candidates) > 0:
                    max_p = max_candidates[0]
                else:
                    continue
                pred_spans.append((min_p, max_p))

            total_obstacles += len(gt_spans)

            for gt_min, gt_max in gt_spans:
                best_error = float('inf')
                for pred_min, pred_max in pred_spans:
                    gt_width = (gt_max - gt_min + 360) % 360
                    pred_width = (pred_max - pred_min + 360) % 360
                    error = abs(pred_width - gt_width)
                    best_error = min(best_error, error)
                total_absolute_error.append(best_error)
                if best_error <= tolerance:
                    matched_obstacles += 1
            if batch_idx == 7:
                print(f"Batch {batch_idx} Sample {b+1}/{batch_size} Predicted Spans: {pred_spans}")
                print(f"Batch {batch_idx} Sample {b+1}/{batch_size} Ground Truth Spans: {gt_spans}")
            
        # Map gate_stats keys to gating_info keys
        key_map = {
            'min_gate_avg': 'min_gate',
            'max_gate_avg': 'max_gate',
            'uncertainty_avg': 'uncertainty',
            'confidence_avg': 'confidence',
        }
        for k in gate_stats:
            gate_stats[k] += gating_info[key_map[k]].mean().item()
        num_batches += 1

    avg_error = np.mean(total_absolute_error)
    obstacle_accuracy = 100.0 * matched_obstacles / total_obstacles

    # Logging
    print(f'Total GT Obstacles: {total_obstacles}')
    print(f'Matched Obstacles (±{tolerance}° width): {matched_obstacles}')
    print(f'Obstacle Detection Accuracy: {obstacle_accuracy:.2f}%')
    print(f'Average Width Error: {avg_error:.2f} degrees')

    print('Gating Statistics:')
    for k in gate_stats:
        print(f'  {k.replace("_", " ").title()}: {gate_stats[k] / num_batches:.3f}')

    # Plotting
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 3, 1)
    plt.plot(range(num_epochs), loss_history, label='Total Loss')
    plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.title('Training Loss'); plt.legend()

    plt.subplot(2, 3, 2)
    for key, values in loss_components.items():
        plt.plot(range(num_epochs), values, label=key)
    plt.xlabel('Epochs'); plt.ylabel('Loss Component'); plt.title('Loss Components'); plt.legend()

    plt.subplot(2, 3, 3)
    gate_names = list(gate_stats.keys())
    gate_values = [v / num_batches for v in gate_stats.values()]
    plt.bar(gate_names, gate_values)
    plt.title('Gating Statistics')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    return obstacle_accuracy, avg_error, gate_stats


def test_model_with_gating_single_sample(
    model, scan, tsdf, device=device, max_spans=3
):
    """
    Run the model on a single LiDAR scan and return predicted (min, max) obstacle spans.
    Uses confidence-based peak detection consistent with test_model_with_gating.
    Ensures that only 0-3 ranges are found.
    """
    model.eval()

    with torch.no_grad():
        model = model.to(device)
        # Prepare scan tensor
        if torch.is_tensor(scan):
            scan_tensor = scan.clone().detach().to(device)
        else:
            scan_tensor = torch.tensor(scan, dtype=torch.float32, device=device)
        if len(scan_tensor.shape) == 1:
            scan_tensor = scan_tensor.unsqueeze(0).unsqueeze(0)
        elif len(scan_tensor.shape) == 2:
            scan_tensor = scan_tensor.unsqueeze(0)
        # Prepare tsdf tensor
        if torch.is_tensor(tsdf):
            tsdf_tensor = tsdf.clone().detach().to(device)
        else:
            tsdf_tensor = torch.tensor(tsdf, dtype=torch.float32, device=device)
        if len(tsdf_tensor.shape) == 3:
            tsdf_tensor = tsdf_tensor.unsqueeze(0)
        elif len(tsdf_tensor.shape) == 2:
            tsdf_tensor = tsdf_tensor.unsqueeze(0).unsqueeze(0)
        elif len(tsdf_tensor.shape) == 1:
            raise ValueError(f"Cannot handle 1D TSDF tensor of shape {tsdf_tensor.shape}. Expected at least 2D.")

        # Forward pass - get predictions as a single dictionary
        predictions = model(scan_tensor, tsdf_tensor)
        
        # Extract outputs (handle both single dict and tuple/list formats)
        if isinstance(predictions, (list, tuple)):
            predictions = predictions[0]
        
        min_pred_logits = predictions['min_logits']
        max_pred_logits = predictions['max_logits']
        count_pred_logits = predictions['count']
        
        # Count prediction: take argmax across class probabilities
        pred_count = torch.argmax(count_pred_logits, dim=1).item()  # Single sample
        
        gating_info = {
            'min_gate': predictions['min_gate'],         
            'max_gate': predictions['max_gate'],         
            'uncertainty': predictions['uncertainty'],   
            'confidence': predictions['confidence']
        }

        # Apply sigmoid to confidence to ensure range [0,1]
        confidence = torch.sigmoid(gating_info['confidence']).detach().cpu().numpy()
        if len(confidence.shape) > 1:
            confidence = confidence[0]  # Single sample
        if len(confidence.shape) > 1:
            confidence = confidence[0]

        # Use top-K confidence peaks based on predicted count
        conf_peaks, _ = find_peaks(confidence, height=0.5, distance=5)
        sorted_peaks = conf_peaks[np.argsort(confidence[conf_peaks])[::-1]]

        predicted_spans = []
        used_mask = np.zeros(360, dtype=bool)
        span_width = 10  # Half-width on each side

        i = 0
        while len(predicted_spans) < pred_count and i < len(sorted_peaks):
            center = sorted_peaks[i]
            i += 1

            min_idx = (center - span_width) % 360
            max_idx = (center + span_width) % 360

            if min_idx <= max_idx:
                span_bins = list(range(min_idx, max_idx + 1))
            else:
                span_bins = list(range(min_idx, 360)) + list(range(0, max_idx + 1))

            # Check for overlap with already used bins
            if any(used_mask[bin_idx] for bin_idx in span_bins):
                continue

            # Mark bins as used
            for bin_idx in span_bins:
                used_mask[bin_idx] = True

            predicted_spans.append((min_idx, max_idx))

        # Fill remaining spans with dummy values if needed
        while len(predicted_spans) < pred_count:
            predicted_spans.append((0, 0))

        # Ensure only 0-max_spans are returned
        if len(predicted_spans) > max_spans:
            # Rank by confidence at peak centers and select top max_spans
            span_scores = []
            for min_idx, max_idx in predicted_spans:
                center = (min_idx + max_idx) // 2
                center_conf = confidence[center % 360]
                width = (max_idx - min_idx + 360) % 360
                span_scores.append(((min_idx, max_idx), center_conf, width))
            
            # Sort by confidence (high to low), then by width (wide to narrow)
            span_scores.sort(key=lambda x: (-x[1], -x[2]))
            predicted_spans = [s[0] for s in span_scores[:max_spans]]

        print(f"[DEBUG] confidence scores: {confidence.round(3)}")
        print(f"[DEBUG] conf_peaks: {conf_peaks}")
        print(f"[DEBUG] sorted_peaks: {sorted_peaks}")
        print(f"[DEBUG] pred_count: {pred_count}")
        print(f"[DEBUG] Raw candidate spans: {predicted_spans}")

        # Prepare gating_info for return (convert to scalars/lists)
        gating_info_output = {}
        for key in ['min_gate', 'max_gate', 'uncertainty', 'confidence']:
            if key in gating_info:
                value = gating_info[key]
                if value.numel() == 1:
                    gating_info_output[key] = value.item()
                else:
                    gating_info_output[key] = value.cpu().numpy().tolist()
        
        return predicted_spans, gating_info_output


# import torch.nn.functional as F
# import numpy as np
# from scipy.signal import find_peaks

# def test_model_with_gating_single_sample(model, scan, tsdf, device=device, max_spans=3):
#     model.eval()
#     model = model.to(device)

#     with torch.no_grad():
#         # Prepare scan tensor
#         scan_tensor = scan if torch.is_tensor(scan) else torch.tensor(scan, dtype=torch.float32)
#         if scan_tensor.dim() == 1:
#             scan_tensor = scan_tensor[None, None, :]  # (1, 1, 360)
#         elif scan_tensor.dim() == 2:
#             scan_tensor = scan_tensor[None, :, :]     # (1, C, 360)
#         scan_tensor = scan_tensor.to(device)

#         # Prepare tsdf tensor
#         tsdf_tensor = tsdf if torch.is_tensor(tsdf) else torch.tensor(tsdf, dtype=torch.float32)
#         if tsdf_tensor.dim() == 2:
#             tsdf_tensor = tsdf_tensor[None, None, :, :]  # (1, 1, H, W)
#         elif tsdf_tensor.dim() == 3:
#             tsdf_tensor = tsdf_tensor[None, :, :, :]     # (1, C, H, W)
#         elif tsdf_tensor.dim() == 1:
#             raise ValueError(f"TSDF tensor shape invalid: {tsdf_tensor.shape}")
#         tsdf_tensor = tsdf_tensor.to(device)

#         # Run forward pass
#         output = model(scan_tensor, tsdf_tensor)
#         count_logits = output['count_logits']  # (1, 4) - for 0,1,2,3 obstacles
#         min_logits_per_obstacle = output['min_logits_per_obstacle']  # (1, 3, 360)
#         max_logits_per_obstacle = output['max_logits_per_obstacle']  # (1, 3, 360)

#         # Get presence probabilities for each obstacle
#         if 'presence_probs_per_obstacle' in output:
#             presence_probs = output['presence_probs_per_obstacle'][0]  # (3, 360)
#         else:
#             # Fallback: extract from individual obstacles
#             presence_probs = torch.stack([
#                 obs['presence_gate'].squeeze() for obs in output['individual_obstacles']
#             ], dim=0)  # (3, 360)

#         # Get confidence probabilities for each obstacle  
#         if 'confidence_probs_per_obstacle' in output:
#             confidence_probs = output['confidence_probs_per_obstacle'][0]  # (3, 360)
#         else:
#             # Fallback: extract from individual obstacles
#             confidence_probs = torch.stack([
#                 obs['confidence'].squeeze() for obs in output['individual_obstacles']
#             ], dim=0)  # (3, 360)

#         batch_idx = 0
#         pred_count = torch.argmax(count_logits, dim=1).item()
#         max_obstacles = min_logits_per_obstacle.shape[1]  # Should be 3

#         # Softmax over angular bins to get probabilities
#         min_probs = F.softmax(min_logits_per_obstacle[batch_idx], dim=1).cpu().numpy()  # (3, 360)
#         max_probs = F.softmax(max_logits_per_obstacle[batch_idx], dim=1).cpu().numpy()  # (3, 360)

#         predicted_spans = []
#         used_bins = np.zeros(360, dtype=bool)

#         print(f"Predicted count: {pred_count}")
#         print("Per-obstacle presence means:", [presence_probs[i].mean().item() for i in range(max_obstacles)])
#         print("Per-obstacle confidence maxes:", [confidence_probs[i].max().item() for i in range(max_obstacles)])

#         # Calculate obstacle scores: combine presence and confidence
#         obstacle_scores = []
#         for i in range(max_obstacles):
#             # Score based on maximum presence and confidence in the obstacle's predicted span
#             min_idx = int(np.argmax(min_probs[i]))
#             max_idx = int(np.argmax(max_probs[i]))
            
#             # Get span indices (handle wraparound)
#             if min_idx <= max_idx:
#                 span_indices = list(range(min_idx, max_idx + 1))
#             else:
#                 span_indices = list(range(min_idx, 360)) + list(range(0, max_idx + 1))
            
#             # Calculate scores over the predicted span
#             span_presence = presence_probs[i][span_indices].mean().item()
#             span_confidence = confidence_probs[i][span_indices].max().item()
            
#             # Combined score
#             score = span_presence * span_confidence
#             obstacle_scores.append((score, i, min_idx, max_idx, span_presence))

#         # Sort by score (highest first)
#         obstacle_scores.sort(key=lambda x: x[0], reverse=True)

#         print("Obstacle scores (score, idx, min, max, presence):")
#         for score, idx, min_idx, max_idx, presence in obstacle_scores:
#             print(f"  Obstacle {idx}: score={score:.3f}, span=({min_idx},{max_idx}), presence={presence:.3f}")

#         # Select top obstacles based on predicted count and score thresholds
#         for score, idx, min_idx, max_idx, span_presence in obstacle_scores:
#             if len(predicted_spans) >= pred_count:
#                 break
                
#             # Skip if score is too low (indicating weak prediction)
#             if score < 0.1:
#                 print(f"  Skipping obstacle {idx}: score {score:.3f} too low")
#                 continue
                
#             # Skip if presence is too low
#             if span_presence < 0.2:
#                 print(f"  Skipping obstacle {idx}: presence {span_presence:.3f} too low")
#                 continue

#             # Check for overlap with already selected spans
#             if min_idx <= max_idx:
#                 span = list(range(min_idx, max_idx + 1))
#             else:
#                 span = list(range(min_idx, 360)) + list(range(0, max_idx + 1))

#             # Allow some overlap but not complete overlap
#             overlap = np.sum(used_bins[span])
#             if overlap > len(span) * 0.5:  # More than 50% overlap
#                 print(f"  Skipping obstacle {idx}: too much overlap ({overlap}/{len(span)})")
#                 continue

#             # Accept this obstacle
#             used_bins[span] = True
#             predicted_spans.append((min_idx, max_idx))
#             print(f"  Accepted obstacle {idx}: span=({min_idx},{max_idx}), score={score:.3f}")

#         # Pad with (0,0) to reach max_spans
#         while len(predicted_spans) < max_spans:
#             predicted_spans.append((0, 0))

#         # Collect additional gating info for debugging
#         gating_info = {}
#         for key in ['min_gate', 'max_gate', 'uncertainty', 'confidence']:
#             if key in output:
#                 val = output[key]
#                 if val.numel() == 1:
#                     gating_info[key] = val.item()
#                 else:
#                     gating_info[key] = val[0].cpu().numpy().tolist()  # Take first batch element

#         # Add per-obstacle information to gating_info
#         gating_info['per_obstacle_scores'] = [score for score, _, _, _, _ in obstacle_scores]
#         gating_info['predicted_count'] = pred_count
        
#         return predicted_spans[:max_spans], gating_info


def test_model_adaptive_threshold(model, scan, tsdf, device=device, debug=True):
    """
    Test model with per-obstacle predictions, ensuring only 0-3 spans are found.
    """
    spans, gating_info = test_model_with_gating_single_sample(
        model, scan, tsdf, device=device)
    
    if debug:
        print(f"Adaptive threshold result: {len([s for s in spans if s != (0, 0)])} spans found")
        for i, span in enumerate(spans):
            if span != (0, 0):  # Only show non-empty spans
                width = (span[1] - span[0] + 360) % 360
                print(f"  Span {i+1}: angles {span[0]}-{span[1]} (width: {width}°)")
        
        print(f"Predicted obstacle count: {gating_info.get('predicted_count', 'Unknown')}")
        print(f"Per-obstacle scores: {[f'{s:.3f}' for s in gating_info.get('per_obstacle_scores', [])]}")
    
    return spans, gating_info


def test_model_with_gating_single_sample_fixed_count(resnet, distances, tsdf, device=torch.device('cuda')):
    import numpy as np
    import torch
    import matplotlib.pyplot as plt

    resnet.eval()
    with torch.no_grad():
        # Prepare input
        if distances.ndim == 1:
            distances = distances.unsqueeze(0).unsqueeze(0)
        elif distances.ndim == 2 and distances.shape[0] == 1:
            distances = distances.unsqueeze(1)

        if tsdf.ndim == 3:
            tsdf = tsdf.unsqueeze(0)

        distances = distances.to(device)
        tsdf = tsdf.to(device)

        # Run model
        predictions = resnet(distances, tsdf)
        min_logits_all = predictions['min_logits'][0].detach().cpu().numpy()  # (3, 360)
        max_logits_all = predictions['max_logits'][0].detach().cpu().numpy()  # (3, 360)
        confidence_all = torch.sigmoid(predictions['confidence'][0]).cpu().numpy()  # (3, 360)

        pred_spans = []
        used_bins = np.zeros(360, dtype=bool)

        for obs_idx in range(3):
            conf_scores = confidence_all[obs_idx]
            min_logits = min_logits_all[obs_idx]
            max_logits = max_logits_all[obs_idx]

            peak = np.argmax(conf_scores)
            search_range = [(peak + i) % 360 for i in range(-30, 31)]

            min_idx = max(search_range, key=lambda i: min_logits[i])
            max_idx = max(search_range, key=lambda i: max_logits[i])

            span_width = (max_idx - min_idx + 360) % 360
            if span_width < 1:
                print(f"[WARN] Skipped span too narrow: obs {obs_idx}")
                continue

            span_bins = list(range(min_idx, max_idx + 1)) if min_idx <= max_idx else \
                        list(range(min_idx, 360)) + list(range(0, max_idx + 1))
            if any(used_bins[i] for i in span_bins):
                print(f"[WARN] Skipped span overlapping: obs {obs_idx}")
                continue

            for i in span_bins:
                used_bins[i] = True

            pred_spans.append((min_idx, max_idx))

        # Fallback: add more spans if < 3
        if len(pred_spans) < 3:
            print(f"[INFO] Fallback triggered: {3 - len(pred_spans)} spans missing")
            avg_conf = confidence_all.mean(axis=0)
            fallback_peaks = np.argsort(avg_conf)[::-1]

            for peak in fallback_peaks:
                if len(pred_spans) >= 3:
                    break
                if used_bins[peak]:
                    continue

                min_idx = (peak - 5) % 360
                max_idx = (peak + 5) % 360

                span_bins = list(range(min_idx, max_idx + 1)) if min_idx <= max_idx else \
                            list(range(min_idx, 360)) + list(range(0, max_idx + 1))

                if any(used_bins[i] for i in span_bins):
                    continue

                for i in span_bins:
                    used_bins[i] = True

                pred_spans.append((min_idx, max_idx))

        # Final fallback: fill with (0, 10)
        while len(pred_spans) < 3:
            pred_spans.append((0, 10))

        print(f"[RESULT] Final predicted spans: {pred_spans}")
        return pred_spans







def test_model_count_single(model, scan, tsdf, device=device, debug=True):
    """
    Test model with per-obstacle predictions, ensuring only 0-3 spans are found.
    """
    spans, gating_info = test_model_with_gating_single_sample_fixed_count(
        model, scan, tsdf, device=device)
    
    if debug:
        print(f"Adaptive threshold result: {len([s for s in spans if s != (0, 0)])} spans found")
        for i, span in enumerate(spans):
            if span != (0, 0):  # Only show non-empty spans
                width = (span[1] - span[0] + 360) % 360
                print(f"  Span {i+1}: angles {span[0]}-{span[1]} (width: {width}°)")
        
        print(f"Predicted obstacle count: {gating_info.get('predicted_count', 'Unknown')}")
        print(f"Per-obstacle scores: {[f'{s:.3f}' for s in gating_info.get('per_obstacle_scores', [])]}")
    
    return spans, gating_info


def debug_model_output(model, scan, tsdf, device=device):
    """
    Debug function to examine model output structure
    """
    model.eval()
    model = model.to(device)
    
    with torch.no_grad():
        # Prepare inputs
        scan_tensor = torch.tensor(scan, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
        tsdf_tensor = torch.tensor(tsdf, dtype=torch.float32, device=device).unsqueeze(0)
        
        print("=== DEBUG MODEL OUTPUT ===")
        print(f"Input shapes: scan={scan_tensor.shape}, tsdf={tsdf_tensor.shape}")
        
        # Run forward pass
        outputs = model(scan_tensor, tsdf_tensor)
        
        print(f"Model output type: {type(outputs)}")
        print(f"Model output length: {len(outputs) if isinstance(outputs, (list, tuple)) else 'Not a sequence'}")
        
        if isinstance(outputs, (list, tuple)):
            for i, output in enumerate(outputs):
                print(f"Output {i}: type={type(output)}")
                if isinstance(output, dict):
                    print(f"  Keys: {list(output.keys())}")
                    for key, value in output.items():
                        if hasattr(value, 'shape'):
                            print(f"    {key}: shape={value.shape}, device={value.device}")
                        else:
                            print(f"    {key}: {type(value)}")
                elif isinstance(output, (list, tuple)):
                    print(f"  List/tuple of length {len(output)}")
                    for j, item in enumerate(output):
                        if hasattr(item, 'shape'):
                            print(f"    [{j}]: shape={item.shape}")
                        else:
                            print(f"    [{j}]: {type(item)}")
        
        return outputs

def test_model_no_gating_single_sample(
    model, scan, tsdf, device=device, max_spans=3
):
    """
    Run model on single LiDAR scan + TSDF.
    Predict up to max_spans obstacle ranges based on confidence peaks.
    Returns predicted spans [(min_idx, max_idx), ...] and gating info.
    """
    model.eval()
    with torch.no_grad():
        model = model.to(device)

        # Prepare scan tensor [1,1,length]
        if not torch.is_tensor(scan):
            scan_tensor = torch.tensor(scan, dtype=torch.float32, device=device)
        else:
            scan_tensor = scan.clone().detach().to(device)
        if scan_tensor.dim() == 1:
            scan_tensor = scan_tensor.unsqueeze(0).unsqueeze(0)
        elif scan_tensor.dim() == 2:
            scan_tensor = scan_tensor.unsqueeze(0)

        # Prepare TSDF tensor [1,C,H,W]
        if not torch.is_tensor(tsdf):
            tsdf_tensor = torch.tensor(tsdf, dtype=torch.float32, device=device)
        else:
            tsdf_tensor = tsdf.clone().detach().to(device)
        if tsdf_tensor.dim() == 2:
            tsdf_tensor = tsdf_tensor.unsqueeze(0).unsqueeze(0)
        elif tsdf_tensor.dim() == 3:
            tsdf_tensor = tsdf_tensor.unsqueeze(0)
        elif tsdf_tensor.dim() == 1:
            raise ValueError("TSDF tensor must be at least 2D.")

        # Forward pass
        preds = model(scan_tensor, tsdf_tensor)
        if isinstance(preds, (list, tuple)):
            preds = preds[0]

        # Get confidence logits and apply sigmoid
        confidence_logits = preds['confidence']
        confidence = torch.sigmoid(confidence_logits).cpu().numpy()
        if confidence.ndim > 1:
            confidence = confidence[0]
        if confidence.ndim > 1:
            confidence = confidence[0]

        # Peak detection on confidence scores
        conf_peaks, _ = find_peaks(confidence, height=0.5, distance=5)
        sorted_peaks = conf_peaks[np.argsort(confidence[conf_peaks])[::-1]]

        predicted_spans = []
        used_mask = np.zeros(360, dtype=bool)
        span_half_width = 10
        pred_count = max_spans  # max_spans = max number of predicted obstacles

        i = 0
        while len(predicted_spans) < pred_count and i < len(sorted_peaks):
            center = sorted_peaks[i]
            i += 1

            min_idx = (center - span_half_width) % 360
            max_idx = (center + span_half_width) % 360

            # bins spanned by obstacle
            if min_idx <= max_idx:
                bins = list(range(min_idx, max_idx + 1))
            else:
                bins = list(range(min_idx, 360)) + list(range(0, max_idx + 1))

            if any(used_mask[b] for b in bins):
                continue

            for b in bins:
                used_mask[b] = True

            predicted_spans.append((min_idx, max_idx))

        # Pad with dummy spans if fewer spans predicted than max_spans
        while len(predicted_spans) < max_spans:
            predicted_spans.append((0, 0))

        # Prepare gating info output dict, convert tensors to python lists/scalars
        gating_info = {k: preds[k] for k in ['min_gate', 'max_gate', 'uncertainty', 'confidence'] if k in preds}
        gating_info_output = {}
        for k, v in gating_info.items():
            if v.numel() == 1:
                gating_info_output[k] = v.item()
            else:
                gating_info_output[k] = v.cpu().tolist()

        return predicted_spans, gating_info_output

import numpy as np
import torch

def emergency_debug_and_fix(model, scan, tsdf, device=None):
    """Emergency debug to see what's actually happening"""
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.eval()
    model = model.to(device)
    
    with torch.no_grad():
        # Prepare inputs
        scan_tensor = scan if torch.is_tensor(scan) else torch.tensor(scan, dtype=torch.float32)
        if scan_tensor.dim() == 1:
            scan_tensor = scan_tensor[None, None, :]
        elif scan_tensor.dim() == 2:
            scan_tensor = scan_tensor[None, :, :]
        scan_tensor = scan_tensor.to(device)
        
        tsdf_tensor = tsdf if torch.is_tensor(tsdf) else torch.tensor(tsdf, dtype=torch.float32)
        if tsdf_tensor.dim() == 2:
            tsdf_tensor = tsdf_tensor[None, None, :, :]
        elif tsdf_tensor.dim() == 3:
            tsdf_tensor = tsdf_tensor[None, :, :, :]
        tsdf_tensor = tsdf_tensor.to(device)
        
        # Get model outputs
        output = model(scan_tensor, tsdf_tensor)
        min_logits = output['min_logits'][0].cpu().numpy()  # (3, 360)
        max_logits = output['max_logits'][0].cpu().numpy()  # (3, 360)
        confidence_logits = output['confidence'][0].cpu().numpy()  # (3, 360)
        
        confidence_probs = 1 / (1 + np.exp(-confidence_logits))
        
        print("🚨 EMERGENCY DIAGNOSIS 🚨")
        print("=" * 50)
        
        # Check 1: Are the outputs completely flat?
        print("\n1. CHECKING FOR FLAT OUTPUTS:")
        for i in range(3):
            min_std = np.std(min_logits[i])
            max_std = np.std(max_logits[i])
            conf_std = np.std(confidence_probs[i])
            
            print(f"   Obstacle {i}: min_std={min_std:.6f}, max_std={max_std:.6f}, conf_std={conf_std:.6f}")
            
            if min_std < 0.001 and max_std < 0.001:
                print(f"   ❌ Obstacle {i}: COMPLETELY FLAT - Model not trained!")
            elif min_std < 0.01 and max_std < 0.01:
                print(f"   ⚠️  Obstacle {i}: Nearly flat - Poor training")
            else:
                print(f"   ✅ Obstacle {i}: Has variation")
        
        # Check 2: What's the actual confidence distribution?
        print("\n2. CONFIDENCE ANALYSIS:")
        for i in range(3):
            max_conf = confidence_probs[i].max()
            max_conf_angle = np.argmax(confidence_probs[i])
            above_50 = np.sum(confidence_probs[i] > 0.5)
            above_30 = np.sum(confidence_probs[i] > 0.3)
            
            print(f"   Obstacle {i}:")
            print(f"     Max confidence: {max_conf:.4f} at angle {max_conf_angle}")
            print(f"     Angles > 0.5 confidence: {above_50}")
            print(f"     Angles > 0.3 confidence: {above_30}")
            
            if above_30 > 0:
                high_conf_angles = np.where(confidence_probs[i] > 0.3)[0]
                print(f"     High confidence angles: {high_conf_angles[:10]}...")  # Show first 10
        
        # Check 3: Are min/max predictions reasonable?
        print("\n3. MIN/MAX PREDICTION ANALYSIS:")
        for i in range(3):
            min_pred = np.argmax(min_logits[i])
            max_pred = np.argmax(max_logits[i])
            min_val = min_logits[i][min_pred]
            max_val = max_logits[i][max_pred]
            
            print(f"   Obstacle {i}:")
            print(f"     Min prediction: angle {min_pred} (logit: {min_val:.4f})")
            print(f"     Max prediction: angle {max_pred} (logit: {max_val:.4f})")
            
            if min_pred == max_pred:
                print(f"     ❌ Same angle for min/max!")
            
            # Check top 3 predictions
            top3_min = np.argsort(min_logits[i])[-3:][::-1]
            top3_max = np.argsort(max_logits[i])[-3:][::-1]
            print(f"     Top 3 min angles: {top3_min} (values: {min_logits[i][top3_min]})")
            print(f"     Top 3 max angles: {top3_max} (values: {max_logits[i][top3_max]})")

def emergency_fix_prediction(model, scan, tsdf, device=None, force_detection=True):
    """Emergency fix that forces detection even with poor model"""
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.eval()
    model = model.to(device)
    
    with torch.no_grad():
        # Prepare inputs (same as above)
        scan_tensor = scan if torch.is_tensor(scan) else torch.tensor(scan, dtype=torch.float32)
        if scan_tensor.dim() == 1:
            scan_tensor = scan_tensor[None, None, :]
        elif scan_tensor.dim() == 2:
            scan_tensor = scan_tensor[None, :, :]
        scan_tensor = scan_tensor.to(device)
        
        tsdf_tensor = tsdf if torch.is_tensor(tsdf) else torch.tensor(tsdf, dtype=torch.float32)
        if tsdf_tensor.dim() == 2:
            tsdf_tensor = tsdf_tensor[None, None, :, :]
        elif tsdf_tensor.dim() == 3:
            tsdf_tensor = tsdf_tensor[None, :, :, :]
        tsdf_tensor = tsdf_tensor.to(device)
        
        output = model(scan_tensor, tsdf_tensor)
        min_logits = output['min_logits'][0].cpu().numpy()
        max_logits = output['max_logits'][0].cpu().numpy()
        confidence_logits = output['confidence'][0].cpu().numpy()
        
        confidence_probs = 1 / (1 + np.exp(-confidence_logits))
        
        pred_spans = []
        confs = []
        
        print("\n🔧 EMERGENCY FIX APPLIED:")
        
        for i in range(3):
            max_conf = confidence_probs[i].max()
            max_conf_angle = np.argmax(confidence_probs[i])
            
            # Strategy 1: If model is completely broken, use scan data directly
            if np.std(confidence_probs[i]) < 0.01 and force_detection:
                print(f"   Obstacle {i}: Model broken, using scan-based detection")
                
                # Use the scan data to find obstacles (basic peak detection)
                scan_np = scan.cpu().numpy() if torch.is_tensor(scan) else scan
                
                # Find local maxima in scan (obstacles appear as peaks)
                from scipy.signal import find_peaks
                peaks, _ = find_peaks(scan_np, height=np.mean(scan_np) + np.std(scan_np))
                
                if len(peaks) > i:
                    center = peaks[i]
                    span_start = max(0, center - 10)  # 20-degree span
                    span_end = min(359, center + 10)
                    pred_spans.append((span_start, span_end))
                    print(f"     Scan-based span: ({span_start}, {span_end})")
                else:
                    pred_spans.append((0, 0))
                    print(f"     No obstacle found in scan")
                    
                confs.append(0.5)  # Default confidence
                continue
            
            # Strategy 2: Use confidence threshold but be more lenient
            threshold = 0.3  # Lower threshold
            high_conf_mask = confidence_probs[i] > threshold
            
            if np.any(high_conf_mask):
                high_conf_angles = np.where(high_conf_mask)[0]
                span_start = int(high_conf_angles.min())
                span_end = int(high_conf_angles.max())
                
                # Ensure minimum span size
                if span_end - span_start < 3:
                    center = (span_start + span_end) // 2
                    span_start = max(0, center - 1)
                    span_end = min(359, center + 2)
                
                pred_spans.append((span_start, span_end))
                print(f"   Obstacle {i}: confidence-based span ({span_start}, {span_end}), conf={max_conf:.3f}")
            
            # Strategy 3: Fall back to top confidence region
            else:
                if max_conf > 0.1:  # Very low threshold
                    span_start = max(0, max_conf_angle - 2)
                    span_end = min(359, max_conf_angle + 2)
                    pred_spans.append((span_start, span_end))
                    print(f"   Obstacle {i}: fallback span ({span_start}, {span_end}), conf={max_conf:.3f}")
                else:
                    pred_spans.append((0, 0))
                    print(f"   Obstacle {i}: no detection, conf={max_conf:.3f}")
            
            confs.append(max_conf)
        
        return pred_spans, confs

# Quick test function
def quick_model_test(model, scan, tsdf):
    """Quick test to see if your model works at all"""
    print("🧪 QUICK MODEL TEST:")
    
    # Run emergency debug
    emergency_debug_and_fix(model, scan, tsdf)
    
    # Try emergency fix
    spans, confs = emergency_fix_prediction(model, scan, tsdf)
    
    print(f"\n📊 RESULTS:")
    print(f"Predicted spans: {spans}")
    print(f"Confidences: {[f'{c:.3f}' for c in confs]}")
    
    # Check if we got reasonable results
    non_zero_spans = sum(1 for s in spans if s != (0, 0))
    print(f"Non-zero spans detected: {non_zero_spans}/3")
    
    if non_zero_spans == 0:
        print("\n❌ CRITICAL: Model is completely broken")
        print("Recommendations:")
        print("1. Check if model weights loaded correctly")
        print("2. Verify input preprocessing")
        print("3. Model needs retraining from scratch")
    elif non_zero_spans < 3:
        print(f"\n⚠️  Model partially working ({non_zero_spans}/3 obstacles)")
        print("Recommendations:")
        print("1. Lower confidence threshold")
        print("2. Train for more epochs")
        print("3. Check loss function weights")
    else:
        print("\n✅ Model detecting obstacles - check span accuracy")
    
    return spans, confs

# RUN THIS FIRST:
# spans, confs = quick_model_test(model, scan, tsdf)


