import os
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from os import path
from pseudoslam.envs.simulator.pseudoSlam import pseudoSlam
import sys, select, termios, tty
import random
import torch
from torch.utils.data import DataLoader
from scipy import ndimage
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from model_testing import test_model_with_gating_loader, test_model_with_gating_single_sample, debug_model_output, test_model_adaptive_threshold, test_model_with_gating_single_sample_fixed_count
from model_testing import quick_model_test, emergency_debug_and_fix, emergency_fix_prediction
from pseudoslam.envs.resnet_2obs import test_model_with_improved_processing_single, LidarTSDFDataset

try:
    from pseudoslam.envs.resnet_2obs import ImprovedResNetNoGating, lidar_to_tsdf_obstacles
except ModuleNotFoundError:
    # Fallback: try importing from local directory if running as a script
    from resnet_2obs import ImprovedResNetNoGating, lidar_to_tsdf_obstacles

# Try importing ResNet from the correct location; adjust as needed if the file is elsewhere
"""try:
    from pseudoslam.envs.resnet import ResNet
except ModuleNotFoundError:
    # Fallback: try importing from local directory if running as a script
    from resnet import ResNet

try:
    from pseudoslam.envs.resnet_tsdf import ResNetWithTSDF
except ModuleNotFoundError:
    # Fallback: try importing from local directory if running as a script
    from resnet_tsdf import ResNetWithTSDF """

"""try:
    from resnet_tsdf_save import ImprovedResNetWithTSDF
except ModuleNotFoundError:
    # Fallback: try importing from local directory if running as a script
    from resnet_tsdf_save import ImprovedResNetWithTSDF

from resnet_tsdf import lidar_to_tsdf_obstacles

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") """

"""base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, "data")
model_path = os.path.join(data_dir, "resnet_lidar_model.pth")

resnet = ResNet()
resnet.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
resnet.eval()"""

# try:
#     from pseudoslam.envs.resnet_gated import ImprovedResNetWithGating, lidar_to_tsdf_obstacles,generate_multi_obstacle_targets
# except ModuleNotFoundError:
#     # Fallback: try importing from local directory if running as a script
#     from resnet_gated import ImprovedResNetWithGating

# try:
#     from pseudoslam.envs.resnet_multi_gated import ImprovedResNetWithGating, lidar_to_tsdf_obstacles, generate_multi_obstacle_targets
# except ModuleNotFoundError:
#     # Fallback: try importing from local directory if running as a script
#     from resnet_multi_gated import ImprovedResNetWithGating

# try:
#     from pseudoslam.envs.resnet_counting import ImprovedResNetWithGating, lidar_to_tsdf_obstacles, generate_multi_obstacle_targets
# except ModuleNotFoundError:
#     # Fallback: try importing from local directory if running as a script
#     from resnet_counting import ImprovedResNetWithGating

# try: 
#     from pseudoslam.envs.resnet_3obs import ImprovedResNetNoGating, lidar_to_tsdf_obstacles
# except ModuleNotFoundError:
#     # Fallback: try importing from local directory if running as a script
#     from resnet_3obs import ImprovedResNetNoGating, lidar_to_tsdf_obstacles


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, "data")
model_path = os.path.join(data_dir, "resnet_2.pth")

resnet = ImprovedResNetNoGating()
resnet.to(device)
# Load the model state dictionary, ignoring size mismatches
checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

# Remove the mismatched tsdf_proj parameters
if 'tsdf_proj.weight' in checkpoint:
    print("Removing mismatched tsdf_proj.weight")
    del checkpoint['tsdf_proj.weight']
if 'tsdf_proj.bias' in checkpoint:
    print("Removing mismatched tsdf_proj.bias")
    del checkpoint['tsdf_proj.bias']

# Remove fusion_conv1 and fusion_conv2 weights if their shapes mismatch
for key in ['fusion_conv1.weight', 'fusion_conv2.weight']:
    if key in checkpoint:
        print(f"Removing mismatched {key}")
        del checkpoint[key]

# Load the modified state dict
resnet.load_state_dict(checkpoint, strict=False)
resnet.eval()


"""resnet = ImprovedResNetWithTSDF()
resnet.to(device)obs_b = obs_a
resnet.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')), strict=False)
resnet.eval()"""

moveBindings = {'w':(1,0),
                'a':(0,1),
                'd':(0,-1),
                's':(-1,0)}

def getKey():
    settings = termios.tcgetattr(sys.stdin)
    tty.setraw(sys.stdin.fileno())
    select.select([sys.stdin], [], [], 1)
    key = sys.stdin.read(1)
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key

def getMotion():
    key = getKey()
    if key in moveBindings.keys():
        robot_x = moveBindings[key][0]
        th = moveBindings[key][1]
        return robot_x,th
    if (key in ['q', 'r', 't', 'e','l', 'm','b','j']):
        return key, key
    else:
        return 0,0

move= {1:'forward', 2:'left', 3:'right'}

def main():
    plt.close('all')  # Close all previous figures to prevent multiple windows
    config_file = path.join(path.dirname(__file__), "config", "config_exploration_obs.yaml")
    sim = pseudoSlam(config_file, obstacle_config=None)
    print("Start simulation.\nThe configuration file locates at {}".format(config_file))
    print("========================================")
    print("Configuration:")
    print("-------------------------------------")
    print("|Robot Radius        |{} pixels      |".format(sim.robotRadius))
    print("|Forward Step Length |{} pixels      |".format(sim.stepLength_linear))
    print("|Rotation Step Length|{:.2} rad      |".format(sim.stepLength_angular))
    print("|Laser Range         |{} pixels    |".format(sim.laser_range))
    print("|Laser FoV           |{:.3} rad      |".format(sim.laser_fov))
    print("|Map Resolution      |{} pixel/meter|".format(sim.m2p))
    print("|Laser Noise         |{}           |".format(sim.laser_noiseSigma))
    print("|SLAM Error          |{}, {:.3}     |".format(sim.slamErr_linear, sim.slamErr_angular))
    print("|Obstacle Number     |{}             |".format(sim.obs_num))
    print("-------------------------------------")
    print("Action command: ")
    print("-------------------------------------")
    print("w: move forward.")
    print("a: rotate clockwise.")
    print("d: rotate anti-clockwise.")
    print("r: reset the environment")
    print("q: stop simulation and quit.")
    print("e: print the laser scan data.")
    print("-------------------------------------")

    fig_map = plt.gcf()

    # Create the laser scan figure once
    fig_laser, ax_laser = plt.subplots(figsize=(8, 6))
    plt.ion()  # Enable interactive mode

    counter = 0
    while 1:
        counter += 1
        slamMap= sim.get_state()
        pose = sim.get_pose()  # pose is not used, but may be useful for debugging

        plt.figure(fig_map.number)
        plt.clf()
        plt.imshow(slamMap,cmap="gray")
        # if counter % 1 == 0:
            # plt.savefig(f"test_{counter}.png")
        plt.draw()
        plt.pause(0.001)

        v,w= getMotion()
        if v==1:
            sim.moveRobot("forward")
            # scan = sim.get_laser_scan()
            # saver_lidar = NPYBatchSaver('new_training_lidar')
            # saver_lidar.add_data(scan)
            # labels = build_label_array(sim)
            # saver_labels = NPYBatchSaver('new_training_labels')
            # saver_labels.add_data(labels)
        elif w=='r':
            print("Reset the environment.")
            sim.reset()
        elif w==1:
            sim.moveRobot("left")
        elif w==-1:
            sim.moveRobot("right")
        elif w=='b':
            print(np.unique(sim.slamMap))
            scan = sim.get_laser_scan()
            print("Building the map with laser scan data.")
            sim.plot_laser_scan_on_map(scan)
        elif w=='e':
            print("Laser scan data:")
            scan = sim.get_laser_scan()
            print(scan)
            labels = build_label_array(sim)
            print("Automated labeling:",labels)
            print('Angle range:', sim.angle_range())
            print('Scan', scan)
            print('Labels:', labels)
            plot_laser_scan_live(scan, fig_laser, ax_laser)
            sim.draw_laser_scan_on_map()
            # labels = build_label_array_ml(sim, ml_min, ml_max)
            # Plot ML prediction on the SLAM map
            # sim.draw_laser_scan_on_map_ml(ml_min, ml_max)
            # print("ML Predictions:", f' ML min:{ml_min}, ml_max:{ml_max}')
            # print("Labels from ML:", labels)
            # sim.plot_ml_preds(ml_preds)
        elif w=='q':
            print("Terminate the simulation and quit.")
            break
        elif w=='l':
            loop_pose(sim)
        elif w=='m':
            labels_ml, pred_spans = build_label_array_ml(sim)
            print("Labels from ML:", labels_ml)
            print(f"ML Predicted spans: {pred_spans}")
            sim.draw_laser_scan_on_map_ml(labels_ml)
            plt.imshow(labels_ml[None, :], cmap='gray', aspect='auto')
            plt.title("ML Label Output")
            plt.show()
        elif w=='t':
            # Save the current laser scan and labels to training data
            scan = sim.get_laser_scan()
            saver_lidar = NPYBatchSaver('fix_training_lidar_3obs')
            saver_lidar.add_data(scan)
            labels = build_label_array(sim)
            saver_labels = NPYBatchSaver('fix_training_labels_3obs')
            saver_labels.add_data(labels)
            print("Saved current laser scan and labels to training data.")
        elif w=='j':
            loop_pose_3(sim)
        else:
            continue
    return

def save_data(data, filename):
    """ Append data to a file in the 'data' directory. """
    data_dir = Path(__file__).parent / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist
    file_path = data_dir / f"{filename}.npy"
    
    # Check if file exists and load existing data
    if file_path.exists():
        try:
            existing_data = np.load(file_path)
            
            # Check if existing data is empty or has wrong shape
            if existing_data.size == 0 or len(existing_data.shape) != 2:
                # Start fresh if file is empty or malformed
                updated_data = data.reshape(1, -1)
            else:
                # Append new data as a new row
                updated_data = np.concatenate([existing_data, data.reshape(1, -1)], axis=0)
        except Exception as e:
            print(f"Error loading existing file: {e}. Creating new file.")
            updated_data = data.reshape(1, -1)
    else:
        # File doesn't exist, create it with just this data
        updated_data = data.reshape(1, -1)
    
    # Save the resetupdated data
    np.save(file_path, updated_data)
    print(f"Data saved to {file_path}, shape: {updated_data.shape}")

class NPYBatchSaver:
    def __init__(self, filename, batch_size=1000):
        self.filename = filename
        self.batch_size = batch_size
        self.buffer = []
        self.data_dir = Path(__file__).parent / 'data'
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.file_path = self.data_dir / f"{filename}.npy"
    
    def add_data(self, data):
        """Add data to buffer, flush when batch_size is reached."""
        if data.ndim == 1:
            data = data.reshape(1, -1)
        self.buffer.append(data)
        
        if len(self.buffer) >= self.batch_size:
            self.flush()
    
    def flush(self):
        """Write all buffered data to file."""
        if not self.buffer:
            return
        
        # Concatenate all buffered data at once
        new_data = np.concatenate(self.buffer, axis=0)
        
        if self.file_path.exists():
            try:
                # Use memory mapping for large files
                existing_data = np.load(self.file_path, mmap_mode='r')
                if existing_data.size > 0 and len(existing_data.shape) == 2:
                    # Pre-allocate for better performance
                    total_rows = existing_data.shape[0] + new_data.shape[0]
                    updated_data = np.empty((total_rows, existing_data.shape[1]), 
                                          dtype=existing_data.dtype)
                    updated_data[:existing_data.shape[0]] = existing_data
                    updated_data[existing_data.shape[0]:] = new_data
                else:
                    updated_data = new_data
            except Exception as e:
                print(f"Error loading existing file: {e}. Creating new file.")
                updated_data = new_data
        else:
            updated_data = new_data
        
        # Save with compression for smaller files
        np.save(self.file_path, updated_data)
        self.buffer = []
        print(f"Batch saved to {self.file_path}, shape: {updated_data.shape}")
    
    def __del__(self):
        """Ensure remaining data is saved when object is destroyed."""
        self.flush()

def loop_pose(sim, num_steps=1000, steps_per_action=7, save_dir_lidar='fix_training_lidar', save_dir_labels='fix_training_labels'):
    """
    Simulates a robot moving through the environment to collect LiDAR scans and obstacle label data.
    
    Args:
        sim: The simulator object.
        num_steps: Total number of simulation steps (default: 3000).
        steps_per_action: Number of random actions per step (default: 7).
        save_dir_lidar: Directory to save LiDAR data.
        save_dir_labels: Directory to save label data.
    """
    actions = ["forward", "left", "right"]

    for i in range(num_steps):
        print(f"\nStep {i + 1}/{num_steps}: Executing {steps_per_action} random actions...")

        # Execute multiple random movements
        for _ in range(steps_per_action):
            action = random.choice(actions)
            sim.moveRobot(action)

        # Collect scan and label
        scan = sim.get_laser_scan()
        labels = build_label_array(sim)

        # Save to disk
        save_data(scan, save_dir_lidar)
        save_data(labels, save_dir_labels)

        # Debug print every 10 steps
        if i % 10 == 0:
            print(f"Pose: {sim.get_pose()}")
            print(f"LiDAR Scan: {scan}")
            print(f"Labels: {labels}")

        # Optionally handle map boundaries
        pose = sim.get_pose()
        if isinstance(pose, (tuple, list, np.ndarray)) and len(pose) >= 2:
            y, x = pose[0], pose[1]
        else:
            raise ValueError(f"Invalid pose returned: {pose}")

        # Optional respawn logic
        # Uncomment if needed
        """
        y_min, y_max, x_min, x_max = sim.get_map_boundaries()
        margin = 10
        if y < y_min + margin or y > y_max - margin or x < x_min + margin or x > x_max - margin:
            print(f"Robot near boundary at ({y}, {x}). Respawning...")
            sim._randomizeRobotPose()
        """

    print("\nData collection loop completed.")

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

def loop_pose_3(sim, num_steps=400, steps_per_action=7, save_dir_lidar='fix_testing_lidar_3obs', save_dir_labels='fix_testing_labels_3obs'):
    """
    Simulates a robot moving through the environment to collect LiDAR scans and obstacle label data.
    
    Args:
        sim: The simulator object.
        num_steps: Total number of simulation steps (default: 3000).
        steps_per_action: Number of random actions per step (default: 7).
        save_dir_lidar: Directory to save LiDAR data.
        save_dir_labels: Directory to save label data.
    """
    actions = ["forward", "left", "right"]

    for i in range(num_steps):
        print(f"\nStep {i + 1}/{num_steps}: Executing {steps_per_action} random actions...")

        # Execute multiple random movements
        for _ in range(steps_per_action):
            action = random.choice(actions)
            sim.moveRobot(action)

        # Collect scan and label
        scan = sim.get_laser_scan()
        labels = build_label_array(sim)

        # Only save if there are 3 obstacles
        if get_obstacle_counts(labels).item() == 3:
            save_data(scan, save_dir_lidar)
            save_data(labels, save_dir_labels)

        # Debug print every 10 steps
        if i % 10 == 0:
            print(f"Pose: {sim.get_pose()}")
            print(f"LiDAR Scan: {scan}")
            print(f"Labels: {labels}")

        # Optionally handle map boundaries
        pose = sim.get_pose()
        if isinstance(pose, (tuple, list, np.ndarray)) and len(pose) >= 2:
            y, x = pose[0], pose[1]
        else:
            raise ValueError(f"Invalid pose returned: {pose}")

        # Optional respawn logic
        # Uncomment if needed
        """
        y_min, y_max, x_min, x_max = sim.get_map_boundaries()
        margin = 10
        if y < y_min + margin or y > y_max - margin or x < x_min + margin or x > x_max - margin:
            print(f"Robot near boundary at ({y}, {x}). Respawning...")
            sim._randomizeRobotPose()
        """

    print("\nData collection loop completed.")


def build_label_array(sim):
    """
    Builds a label array of size 360. Angles are marked as 1 *only if*:
    - They fall within a known obstacle range
    - The robot has explored that angle (not in the unknown/black region)
    
    After initial labeling, fills in gaps between edge indices within each obstacle range.
    """
    angle_ranges = sim.angle_range()
    label_array = np.zeros(360, dtype=np.int32)
    scan = sim.get_laser_scan()
    max_range = sim.laser_range if hasattr(sim, "laser_range") else np.max(scan)
    
    if not isinstance(angle_ranges, (list, tuple)):
        angle_ranges = [angle_ranges]
    
    print("Angle ranges:", angle_ranges)
    
    for angle_range_array in angle_ranges:
        print(f"Processing array: {angle_range_array}")
        
        # Handle numpy array of ranges
        if isinstance(angle_range_array, np.ndarray) and angle_range_array.ndim == 2:
            # This is a 2D array where each row is [angle_min, angle_max]
            for angle_range in angle_range_array:
                angle_min, angle_max = angle_range
                angle_min = int(angle_min.item() if hasattr(angle_min, "item") else angle_min)
                angle_max = int(angle_max.item() if hasattr(angle_max, "item") else angle_max)
                
                # Process this specific range
                _process_angle_range(label_array, scan, angle_min, angle_max, max_range)
        else:
            # Handle single range [angle_min, angle_max]
            if len(angle_range_array) != 2:
                continue
            angle_min, angle_max = angle_range_array
            angle_min = int(angle_min.item() if hasattr(angle_min, "item") else angle_min)
            angle_max = int(angle_max.item() if hasattr(angle_max, "item") else angle_max)
            
            # Process this specific range
            _process_angle_range(label_array, scan, angle_min, angle_max, max_range)
    
    return label_array

def _process_angle_range(label_array, scan, angle_min, angle_max, max_range):
    """
    Process a single angle range and fill in gaps between valid detections.
    """
    # Handle wraparound case
    if angle_max < angle_min:
        angles = list(range(angle_min, 360)) + list(range(0, angle_max + 1))
    else:
        angles = list(range(angle_min, angle_max + 1))
    
    print(f"Processing range {angle_min} to {angle_max}, angles: {len(angles)} total")
    
    # First pass: mark all valid detections
    valid_angles = []
    for angle in angles:
        angle = angle % 360
        distance = scan[angle]
        if np.isfinite(distance) and 0 < distance:
            label_array[angle] = 1
            valid_angles.append(angle)
    
    if len(valid_angles) < 2:
        print(f"Not enough valid angles ({len(valid_angles)}) to fill gaps")
        return
    
    # Handle wraparound case for filling
    if angle_max < angle_min:
        # Split into two segments and process each
        segment1 = [a for a in valid_angles if a >= angle_min]
        segment2 = [a for a in valid_angles if a <= angle_max]
        
        if len(segment1) >= 2:
            _fill_gaps_in_segment(label_array, segment1, angle_min, 359)
        if len(segment2) >= 2:
            _fill_gaps_in_segment(label_array, segment2, 0, angle_max)
        
        # Handle connection across wraparound if both segments have valid points
        if segment1 and segment2:
            # Connect the last angle of segment1 with the first angle of segment2
            last_angle = max(segment1)
            first_angle = min(segment2)
            
            # Fill from last_angle to 359, then from 0 to first_angle
            for angle in range(last_angle, 360):
                if angle <= 359:
                    label_array[angle] = 1
            for angle in range(0, first_angle + 1):
                label_array[angle] = 1
    else:
        # Normal case: fill gaps within the range
        _fill_gaps_in_segment(label_array, valid_angles, angle_min, angle_max)

def _fill_gaps_in_segment(label_array, valid_angles, range_min, range_max):
    """
    Fill gaps between valid angles within a continuous segment.
    """
    if len(valid_angles) < 2:
        return
    
    valid_angles = sorted(valid_angles)
    print(f"Filling gaps between valid angles: {valid_angles}")
    
    # Find the leftmost and rightmost valid angles within the range
    leftmost = valid_angles[0]
    rightmost = valid_angles[-1]
    
    # Fill all angles between leftmost and rightmost
    for angle in range(leftmost, rightmost + 1):
        if range_min <= angle <= range_max:
            label_array[angle] = 1
    
    print(f"Filled range {leftmost} to {rightmost}")


# def build_label_array_ml(sim, peak_threshold=0.5):
    # """
    # Runs a single scan through the model and extracts predicted obstacle spans.

    # Args:
    #     sim: Simulation object
    #     peak_threshold: Threshold for considering logits as obstacle edges.

    # Returns:
    #     label_array: numpy array (360,) marking detected obstacle angles as 1
    #     pred_spans: list of (min_idx, max_idx) tuples representing obstacle spans
    # """
    # scan = sim.get_laser_scan()
    # scan_np = np.array(scan).reshape(1, 360)

    # dummy_labels = np.zeros_like(scan_np)
    # dummy_tsdf = lidar_to_tsdf_obstacles(scan_np, dummy_labels)

    # scan_tensor = torch.tensor(scan_np, dtype=torch.float32).unsqueeze(1).to(device)
    # tsdf_tensor = torch.tensor(dummy_tsdf, dtype=torch.float32).unsqueeze(1).to(device)

    # with torch.no_grad():
    #     pred_min_logits, pred_max_logits, gating_info = resnet(scan_tensor, tsdf_tensor)

    # pred_min_scores = torch.sigmoid(pred_min_logits).cpu().numpy().squeeze()
    # pred_max_scores = torch.sigmoid(pred_max_logits).cpu().numpy().squeeze()

    # # Find indices where scores exceed threshold
    # min_candidates = np.where(pred_min_scores >= peak_threshold)[0]
    # max_candidates = np.where(pred_max_scores >= peak_threshold)[0]

    # pred_spans = []
    # for min_idx in min_candidates:
    #     # Find max_candidate that comes after min_idx circularly
    #     max_after = max_candidates[max_candidates >= min_idx]
    #     if len(max_after) == 0 and len(max_candidates) > 0:
    #         max_idx = max_candidates[0]  # wraparound case
    #     elif len(max_after) > 0:
    #         max_idx = max_after[0]
    #     else:
    #         continue
    #     pred_spans.append((min_idx, max_idx))

    # # Build label array from predicted spans
    # label_array = np.zeros(360, dtype=np.int32)
    # for min_idx, max_idx in pred_spans:
    #     if min_idx <= max_idx:
    #         label_array[min_idx:max_idx + 1] = 1
    #     else:
    #         label_array[min_idx:] = 1
    #         label_array[:max_idx + 1] = 1

    # return label_array, pred_spans

def build_label_array_ml(sim):
    
    # sample_lidar = testing_lidar[200]  # Use the first sample for testing
    # sample_tsdf = lidar_to_tsdf_obstacles(np.expand_dims(sample_lidar, axis=0))  # Get corresponding TSDF
    # # Generate dummy labels (all 0s)
    # dummy_labels = np.zeros((1, 360), dtype=np.float32)  # shape: (1, 360)
    # # dummy_labels = np.expand_dims(labels_test[200], axis=0)  # shape: (1, 360)
    # actual_obstacle_spans = generate_multi_obstacle_targets(np.expand_dims(labels_test[200], axis=0))  # Get ground truth
    # single_test = LidarTSDFDataset(
    #     np.expand_dims(sample_lidar, axis=0), 
    #     dummy_labels,  # Dummy labels
    #     np.expand_dims(testing_mask[200], axis=0),
    #     tsdf_cache_path=None)

    # single_test = DataLoader(single_test, batch_size=1, shuffle=False, pin_memory=True, num_workers=4)

    # predicted_spans = test_model_with_improved_processing_single(single_test)

    # print("Predicted obstacle spans:")
    # print(predicted_spans)
    # print("Actual obstacle spans:")
    # print(actual_obstacle_spans)

    scan = np.array(sim.get_laser_scan()).reshape(360)
    tsdf = lidar_to_tsdf_obstacles(scan)
    dummy_labels = np.zeros((1, 360), dtype=np.float32)  # shape: (1, 360)
    testing_mask = np.ones_like(dummy_labels, dtype=np.float32)
    dataset = LidarTSDFDataset(
        np.expand_dims(scan, axis=0), 
        dummy_labels,  # Dummy labels
        np.expand_dims(testing_mask, axis=0),
        tsdf_cache_path=None)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=4)

    predicted_spans = test_model_with_improved_processing_single(dataloader)
    print("Predicted obstacle spans:")
    print(predicted_spans)
    # print("TSDF numpy shape before squeeze:", tsdf.shape)  # Should print (1, 2, 32, 32)
    # tsdf = np.squeeze(tsdf, axis=0)  # Remove leading dim
    # print("TSDF numpy shape after squeeze:", tsdf.shape)   # Should print (2, 32, 32)
    # print("TSDF:", tsdf)
    # scan_tensor = torch.tensor(scan, dtype=torch.float32)  # [1, 1, 360]
    # tsdf_tensor = torch.tensor(tsdf, dtype=torch.float32)  # Should be [2, 360]
    # if tsdf_tensor.ndim == 2:
    #     tsdf_tensor = tsdf_tensor.unsqueeze(0)  # [1, 2, 360]


    # print("distance shape before model:", scan_tensor.shape)
    # print("tsdf shape before model:", tsdf_tensor.shape)

    # spans, gating_info = test_model_adaptive_threshold(
    #     resnet,
    #     scan_tensor,
    #     tsdf_tensor,
    #     device=device
    # )

    # spans = test_model_with_gating_single_sample(
    #     resnet,
    #     scan_tensor,
    #     tsdf_tensor,
    #     device=device
    # )

    # spans, confs = quick_model_test(resnet, scan, tsdf)

    # print("Predicted obstacle spans:", spans)
    # # print("Gating info:", gating_info)

    label_array = np.zeros(360, dtype=np.int32)
    # predicted_spans is like: [[(72, 89), (291, 331)]], so unwrap the outer list first
    for min_idx, max_idx in predicted_spans[0]:  
        # Ensure min_idx and max_idx are integers (in case they are numpy scalars)
        min_idx = int(min_idx)
        max_idx = int(max_idx)

        if min_idx <= max_idx:
            label_array[min_idx:max_idx + 1] = 1
        else:
            label_array[min_idx:] = 1
            label_array[:max_idx + 1] = 1


    # debug_model_output(resnet, sim, tsdf, device=device)
    return label_array, predicted_spans



def run_ml_prediction(sim):
    # Run a single scan through the model and extract predicted obstacle spans.
    scan = sim.get_laser_scan()
    scan_np = np.array(scan).reshape(1, 360)
    dummy_labels = np.zeros_like(scan_np)
    dummy_tsdf = lidar_to_tsdf_obstacles(scan_np, dummy_labels)
    scan_tensor = torch.tensor(scan_np, dtype=torch.float32).to(device)
    tsdf_tensor = torch.tensor(dummy_tsdf, dtype=torch.float32).to(device)

    with torch.no_grad():
        pred_min_logits, pred_max_logits, gating_info = resnet(scan_tensor, tsdf_tensor)

    pred_min_scores = torch.sigmoid(pred_min_logits).cpu().numpy().squeeze()
    pred_max_scores = torch.sigmoid(pred_max_logits).cpu().numpy().squeeze()

    peak_threshold = 0.505
    min_candidates = np.where(pred_min_scores >= peak_threshold)[0]
    max_candidates = np.where(pred_max_scores >= peak_threshold)[0]

    pred_spans = []

    for min_idx in min_candidates:
        # Find the first max after min, wrap around if necessary
        max_after = max_candidates[max_candidates >= min_idx]
        if len(max_after) > 0:
            max_idx = max_after[0]
        elif len(max_candidates) > 0:
            max_idx = max_candidates[0]  # wrap around
        else:
            continue  # no max at all
        
        if min_idx != max_idx:
            pred_spans.append((int(min_idx), int(max_idx)))

    # Optional: merge overlapping or adjacent spans
    merged_spans = []
    for span in sorted(pred_spans, key=lambda x: x[0]):
        if not merged_spans:
            merged_spans.append(span)
        else:
            prev_min, prev_max = merged_spans[-1]
            curr_min, curr_max = span
            # Merge if overlapping or adjacent, allowing wrap-around logic
            if (prev_max >= prev_min and curr_min <= prev_max + 1) or \
               (prev_max < prev_min and (curr_min <= prev_max + 1 or curr_min >= prev_min)):
                merged_spans[-1] = (prev_min, curr_max)
            else:
                merged_spans.append(span)

    # Build label array
    label_array = np.zeros(360, dtype=np.int32)
    for min_idx, max_idx in merged_spans:
        if min_idx <= max_idx:
            label_array[min_idx:max_idx + 1] = 1
        else:
            # wraparound case
            label_array[min_idx:] = 1
            label_array[:max_idx + 1] = 1

    return label_array, merged_spans



def clean_label_array(label_array):
    # Remove isolated 1s (noise)
    label_array = ndimage.binary_opening(label_array, structure=np.ones(3))
    
    # Fill small gaps between 1s
    label_array = ndimage.binary_closing(label_array, structure=np.ones(3))
    
    return label_array.astype(np.int32)

def plot_laser_scan_live(scan, fig, ax):
    # Update the laser scan plot in the same window
    ax.clear()
    angles = range(0, 360)
    ax.plot(angles, scan, marker='o', linestyle='-', color='b', label='Laser Scan')
    ax.set_title('Laser Scan Data')
    ax.set_xlabel('Angle (degrees)')
    ax.set_ylabel('Distance (pixels)')
    ax.set_xticks(np.arange(0, 361, 30))
    ax.set_xlim(0, 360)
    ax.set_ylim(0, max(scan) + 10)
    ax.grid(True)
    ax.legend()
    ax.set_aspect('auto')
    fig.canvas.draw()
    fig.canvas.flush_events()

if __name__=='__main__':
    main()
