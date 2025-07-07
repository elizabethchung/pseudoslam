import numpy as np
from matplotlib import pyplot as plt
from os import path
from pseudoslam.envs.simulator.pseudoSlam import pseudoSlam
import sys, select, termios, tty

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
    if (key in ['q', 'r', 'e']):
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
        if counter % 1 == 0:
            plt.savefig(f"test_{counter}.png")
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
            print(scan)
            print(build_label_array(sim))
            print(sim.angle_range())
            plot_laser_scan_live(scan, fig_laser, ax_laser)
            sim.draw_laser_scan_on_map()
        elif w=='q':
            print("Terminate the simulation and quit.")
            break
        else:
            continue
    return

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
        if scan[angle] < max_range and scan[angle] <= dist_to_obs:
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
