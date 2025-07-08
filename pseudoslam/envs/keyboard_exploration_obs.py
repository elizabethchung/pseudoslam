import os
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from os import path
from pseudoslam.envs.simulator.pseudoSlam import pseudoSlam
import sys, select, termios, tty
import random

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
    if (key in ['q', 'r', 'e','l']):
        return key, key
    else:
        return 0,0

move= {1:'forward', 2:'left', 3:'right'}

def main():
    config_file= path.join(path.dirname(__file__), "config", "config_exploration_obs.yaml")
    sim= pseudoSlam(config_file)
    
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
        pose = sim.get_pose()

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
        elif w=='r':
            print("Reset the environment.")
            sim.reset()
        elif w==1:
            sim.moveRobot("left")
        elif w==-1:
            sim.moveRobot("right")
        elif w=='e':
            print("Laser scan data:")
            scan = sim.get_laser_scan()
            save_data(scan, 'training_lidar')
            print(scan)
            labels = build_label_array(sim)
            print(labels)
            save_data(labels, 'training_labels')
            print(sim.angle_range())
            plot_laser_scan_live(scan, fig_laser, ax_laser)
            sim.draw_laser_scan_on_map()
        elif w=='q':
            print("Terminate the simulation and quit.")
            break
        elif w=='l':
            loop_pose(sim)
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
    
    # Save the updated data
    np.save(file_path, updated_data)
    print(f"Data saved to {file_path}, shape: {updated_data.shape}")

def loop_pose(sim):
    """
    Simulate a user moving around: at each step, pick a random valid action (forward, left, right),
    execute it, and save LIDAR scan and labels. Repeat for 5000 steps.
    """

    actions = ["forward", "left", "right"]
    for i in range(500):
        print(f"Step {i+1}/500: Executing random action...")
        # Randomly choose an action multiple times
        for i in range(5):
            action = random.choice(actions)
            if action == "forward":
                sim.moveRobot("forward")
            elif action == "left":
                sim.moveRobot("left")
            elif action == "right":
                sim.moveRobot("right")
            else:
                print(f"Unknown action: {action}")
                continue
        scan = sim.get_laser_scan()
        save_data(scan, 'testing_lidar')
        labels = build_label_array(sim)
        save_data(labels, 'testing_labels')
        if i % 1 == 0:
            print(f"Step {i}: Action {action}, Pose {sim.get_pose()}")
            print(f"Scan: {scan}")
            print(f"Labels: {labels}")
            # Plot the laser scan live
            """fig_laser, ax_laser = plt.subplots(figsize=(8, 6))
            plot_laser_scan_live(scan, fig_laser, ax_laser)
            # Plot the map with the robot's position
            plt.figure()
            plt.imshow(sim.get_state(), cmap="gray")
            # Add the robot's scan to the map
            sim.draw_laser_scan_on_map()
            plt.title(f"Robot Pose: {sim.get_pose()}")
            plt.pause(15)"""
        # Respawn if the robot has x or y coordinate near map boundary
        pose = sim.get_pose()
        # pose is an array: [y, x, angle]
        if isinstance(pose, (tuple, list, np.ndarray)) and len(pose) >= 2:
            y, x = pose[0], pose[1]
        else:
            raise ValueError(f"sim.get_pose() did not return at least two values: got {pose}")
        # get_map_boundaries returns [y_min, y_max, x_min, x_max]
        boundaries = sim.get_map_boundaries()
        if len(boundaries) != 4:
            raise ValueError(f"sim.get_map_boundaries() did not return 4 values: got {boundaries}")
        y_min, y_max, x_min, x_max = boundaries
        if y < y_min + 10 or y > y_max - 10 or x < x_min + 10 or x > x_max - 10:
            print(f"Robot is near the map boundary at ({y}, {x}). Respawning...")
            # Randomize the robot's position
            sim._randomizeRobotPose()
    print("Loop completed.")

def build_label_array(sim):
    # Use angle_range to get the angles for the laser scan
    angle_min, angle_max = sim.angle_range()
    # Ensure angle_min and angle_max are Python scalars
    angle_min = int(angle_min.item() if hasattr(angle_min, "item") else angle_min)
    angle_max = int(angle_max.item() if hasattr(angle_max, "item") else angle_max)
    # Might go from 340 to 20 degrees, so we need to handle the wrap-around
    if angle_max < angle_min:
        angles = np.arange(angle_min, 360).tolist() + np.arange(0, angle_max + 1).tolist()
    else:
        angles = np.arange(angle_min, angle_max + 1).tolist()
    # Subtract 1 for index and wrap around using modulo 360
    angles = [(angle - 1) % 360 for angle in angles]  # Convert to zero-based index safely
    # Create an array of 360 zeros to start with
    label_array = np.zeros(360, dtype=np.int32)
    # Set the corresponding indices to 1
    scan = sim.get_laser_scan()
    max_range = sim.laser_range if hasattr(sim, "laser_range") else np.max(scan)
    for angle in angles:
        dist_to_obs = sim.distance_to_obstacle(angle)
        # Set label if the scan detects an obstacle (distance less than max range and less than or equal to distance to obstacle)
        if scan[angle] <= dist_to_obs:
            label_array[angle] = 1
    return label_array

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
