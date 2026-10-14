from importlib import util
import numpy as np
import matplotlib.pyplot as plt

sample_lidar4 = np.array([
    100., 100.,  99.,  99.,  99.,  98.,  98.,  98.,  98.,  98.,  98.,  97.,  97.,  97.,
     97.,  97.,  97.,  98.,  98.,  98.,  98.,  98.,  98.,  99.,  99.,  99., 100., 100.,
    101., 101., 102., 102., 103., 103., 104., 105., 105., 106., 107., 108., 109., 110.,
    111., 112., 113., 114., 115., 117., 118., 119., 121., 123., 124., 126., 125., 126.,
    126., 126., 125., 126., 125., 126., 125., 126., 126., 126., 127., 128., 129., 130.,
    131., 132., 134., 136., 137., 132., 129., 131., 134., 125., 127., 127., 126., 126.,
    126., 126., 126., 126., 126., 126., 126., 126., 126., 126., 127., 107., 100., 106.,
    100.,  96.,  95.,  95.,  94.,  93.,  93.,  91.,  91.,  90.,  90.,  89.,  89.,  88.,
     88.,  90.,  91.,  91.,  92.,  94.,  96., 200., 200., 200., 200., 200., 200., 200.,
    200., 200., 200., 200., 200., 200., 200., 200., 200., 200., 200., 200., 200., 200.,
    200., 200., 200., 200., 200., 200., 200., 200., 200., 200., 200., 200., 200., 200.,
    200., 200., 200., 200., 200., 200., 200., 199., 198., 197., 196., 195., 193., 193.,
    191., 191., 189., 188., 187., 185., 184., 183., 181., 180., 178., 177., 175., 174.,
    172., 172., 170., 167., 166., 165., 163., 161., 160., 158., 157., 155., 154., 153.,
    151., 150., 149., 148., 147., 146., 145., 145., 144., 143., 143., 142., 141., 141.,
    140., 140., 140., 140., 139., 138., 139., 138., 138., 138., 138., 138., 137., 137.,
    137., 137., 137., 136., 137., 136., 136., 136., 133., 135., 135., 134., 134., 134.,
    134., 133., 133., 133., 133., 133., 132., 132., 132., 132., 131., 131., 131., 130.,
    130., 130., 130., 130., 128., 128., 128., 128., 127., 128., 127., 127., 127., 126.,
    127., 126., 125., 126., 126., 126., 126., 126., 126., 126., 126., 126., 127., 126.,
    126., 126., 126., 126., 126., 126., 126., 126., 125., 125., 126., 126., 125., 126.,
    126., 125., 126., 125., 126., 125., 126., 126., 125., 126., 126., 126.,  89.,  81.,
     80.,  77.,  76.,  74.,  74.,  73.,  73.,  73.,  73.,  73.,  73.,  72.,  73.,  73.,
     74.,  75.,  76.,  77.,  78.,  81.,  82., 113., 126., 118., 125., 126., 117., 124.,
    121., 121., 120., 118., 117., 115., 114., 113., 112., 111., 110., 109., 108., 107.,
    106., 105., 105., 104., 103., 103., 102., 102., 101., 101.
], dtype=np.float32)

# # visualize_tsdf(torch.tensor(tsdf_result, dtype=torch.float32).unsqueeze(0))
# def lidar_to_tsdf_obstacles(lidar_data, img_dim=32, max_range=200.0, trunc_thresh=3):
#     """
#     Improved LIDAR to TSDF conversion with proper ray casting and correct angle alignment.
#     Returns 2-channel TSDF: [obstacles, free_space]
#     """
#     if lidar_data.ndim == 1:
#         lidar_data = np.expand_dims(lidar_data, axis=0)
    
#     batch_size = lidar_data.shape[0]
#     center = img_dim // 2
#     scale = img_dim / (2 * max_range)  # meters to pixels
    
#     # LiDAR angles: 0° is forward (positive x), increases counterclockwise
#     # To match image coordinates, flip y axis (origin='lower')
#     angles = np.deg2rad(np.arange(360))
    
#     # Pre-compute grid coordinates
#     y_indices, x_indices = np.mgrid[0:img_dim, 0:img_dim]
    
#     # Convert grid indices to world coordinates (centered at origin)
#     x_world_grid = (x_indices - center) / scale
#     y_world_grid = (y_indices - center) / scale
    
#     # Initialize TSDF maps
#     obstacle_tsdf = np.full((batch_size, img_dim, img_dim), trunc_thresh, dtype=np.float32)
#     free_space_tsdf = np.full((batch_size, img_dim, img_dim), trunc_thresh, dtype=np.float32)
    
#     for b in range(batch_size):
#         valid_rays = (lidar_data[b] > 0) & (lidar_data[b] < max_range)
        
#         for i in range(360):
#             if not valid_rays[i]:
#                 continue
                
#             distance = lidar_data[b, i]
#             angle = angles[i]
            
#             # Obstacle endpoint in world coordinates
#             x_obs = distance * np.cos(angle)
#             y_obs = distance * np.sin(angle)
            
#             # Convert to grid coordinates
#             x_obs_grid = center + x_obs * scale
#             y_obs_grid = center + y_obs * scale  # <-- FIX: use +y for image origin='lower'
            
#             # Skip if obstacle is outside grid
#             if x_obs_grid < 0 or x_obs_grid >= img_dim or y_obs_grid < 0 or y_obs_grid >= img_dim:
#                 continue
            
#             # === OBSTACLE TSDF ===
#             # Distance from each grid point to the obstacle point
#             dist_to_obs = np.sqrt((x_indices - x_obs_grid)**2 + (y_indices - y_obs_grid)**2)
#             # TSDF: negative inside obstacle (within 1.5 pixels), positive outside
#             obs_sdf = dist_to_obs - 1.5
#             obs_sdf = np.clip(obs_sdf, -trunc_thresh, trunc_thresh)
#             obstacle_tsdf[b] = np.minimum(obstacle_tsdf[b], obs_sdf)
            
#             # === FREE SPACE TSDF ===
#             # Ray casting: mark all points along the ray as free space
#             ray_steps = int(distance * scale * 1.2)  # Slightly longer than needed
#             if ray_steps > 1:
#                 for step in range(1, ray_steps):
#                     # Current position along the ray
#                     t = step / ray_steps
#                     x_ray = t * x_obs
#                     y_ray = t * y_obs
                    
#                     # Convert to grid coordinates
#                     x_ray_grid = center + x_ray * scale
#                     y_ray_grid = center + y_ray * scale  # <-- FIX: use +y for image origin='lower'
                    
#                     # Skip if outside grid
#                     if x_ray_grid < 0 or x_ray_grid >= img_dim or y_ray_grid < 0 or y_ray_grid >= img_dim:
#                         continue
                    
#                     # Distance from each grid point to this ray point
#                     dist_to_ray = np.sqrt((x_indices - x_ray_grid)**2 + (y_indices - y_ray_grid)**2)
#                     # TSDF: negative inside free space region (within 2 pixels), positive outside
#                     free_sdf = dist_to_ray - 2.0
#                     free_sdf = np.clip(free_sdf, -trunc_thresh, trunc_thresh)
#                     free_space_tsdf[b] = np.minimum(free_space_tsdf[b], free_sdf)

#     # Stack channels: (batch, channels, height, width)
#     result = np.stack([obstacle_tsdf, free_space_tsdf], axis=1)
#     return result

def is_flat(lidar, i, threshold=1.0):
    return abs(lidar[(i+1)%360] + lidar[(i-1)%360] - 2*lidar[i]) < threshold


def lidar_to_tsdf_obstacles(lidar_data, img_dim=32, max_range=125.0, trunc_thresh=3):
    """
    LIDAR to TSDF conversion with obstacle, free-space, and unknown channels.
    Unexplored regions are NOT treated as obstacles.
    """
    if lidar_data.ndim == 1:
        lidar_data = np.expand_dims(lidar_data, axis=0)
    
    batch_size = lidar_data.shape[0]
    center = img_dim // 2
    scale = img_dim / (2 * max_range)  # meters to pixels
    angles = np.deg2rad(np.arange(360))
    
    y_indices, x_indices = np.mgrid[0:img_dim, 0:img_dim]
    
    obstacle_tsdf = np.full((batch_size, img_dim, img_dim), trunc_thresh, dtype=np.float32)
    free_space_tsdf = np.full((batch_size, img_dim, img_dim), trunc_thresh, dtype=np.float32)
    explored_mask = np.zeros((batch_size, img_dim, img_dim), dtype=bool)
    print("Using max_range:", max_range)
    print("LiDAR max:", np.max(lidar_data))
    print("Scale factor:", scale)


    for b in range(batch_size):
        valid_rays = lidar_data[b] > 0  # Keep rays even if they exceed max_range
        
        for i in range(360):
            if not valid_rays[i]:
                continue
            
            distance = lidar_data[b, i]
            angle = angles[i]
            
            # Clamp ray length to max_range
            effective_distance = min(distance, max_range)
            x_end = effective_distance * np.cos(angle)
            y_end = effective_distance * np.sin(angle)
            x_end_grid = center + x_end * scale
            y_end_grid = center + y_end * scale
            
            if 0 <= x_end_grid < img_dim and 0 <= y_end_grid < img_dim:
                # Mark endpoint as explored even if it's just the end of the ray
                explored_mask[b][int(y_end_grid), int(x_end_grid)] = True

                # Cast ray regardless of hit/miss
                ray_steps = int(effective_distance * scale * 1.2)
                if ray_steps > 1:
                    for step in range(1, ray_steps):
                        t = step / ray_steps
                        x_ray = t * x_end
                        y_ray = t * y_end
                        x_ray_grid = center + x_ray * scale
                        y_ray_grid = center + y_ray * scale
                        
                        if 0 <= x_ray_grid < img_dim and 0 <= y_ray_grid < img_dim:
                            dist_to_ray = np.sqrt((x_indices - x_ray_grid)**2 + (y_indices - y_ray_grid)**2)
                            free_sdf = dist_to_ray - 2.0
                            free_sdf = np.clip(free_sdf, -trunc_thresh, trunc_thresh)
                            free_space_tsdf[b] = np.minimum(free_space_tsdf[b], free_sdf)
                            explored_mask[b][dist_to_ray <= 2.0] = True

            # Only mark obstacle if distance < max_range
            if distance < max_range:
                def is_flat(lidar, idx, threshold=1.3):
                    prev = lidar[(idx - 1) % 360]
                    next = lidar[(idx + 1) % 360]
                    return abs(prev + next - 2 * lidar[idx]) < threshold
                
                if is_flat(lidar_data[b], i):
                    continue

                x_obs = distance * np.cos(angle)
                y_obs = distance * np.sin(angle)
                x_obs_grid = center + x_obs * scale
                y_obs_grid = center + y_obs * scale
                
                if 0 <= x_obs_grid < img_dim and 0 <= y_obs_grid < img_dim:
                    dist_to_obs = np.sqrt((x_indices - x_obs_grid)**2 + (y_indices - y_obs_grid)**2)
                    obs_sdf = dist_to_obs - 1.5
                    obs_sdf = np.clip(obs_sdf, -trunc_thresh, trunc_thresh)
                    obstacle_tsdf[b] = np.minimum(obstacle_tsdf[b], obs_sdf)
                    explored_mask[b][dist_to_obs <= 1.5] = True

    unknown_channel = (~explored_mask).astype(np.float32)
    # Final masking to ensure clean outputs
    obstacle_tsdf = np.where(explored_mask, obstacle_tsdf, 0)
    free_space_tsdf = np.where(explored_mask, free_space_tsdf, 0)
    result = np.stack([obstacle_tsdf, free_space_tsdf, unknown_channel], axis=1)
    return result



# def visualize_tsdf_with_lidar(lidar_data, tsdf_result, img_dim=32, max_range=200.0):
#     """
#     Create a visualization showing LiDAR points overlaid on TSDF
#     """

    
#     center = img_dim // 2
#     scale = img_dim / (2 * max_range)
#     angles = np.deg2rad(np.arange(360))
    
#     fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
#     # Plot obstacle TSDF
#     im1 = axes[0].imshow(tsdf_result[0, 0], cmap='RdYlBu_r', vmin=-3, vmax=3)
#     axes[0].set_title('Obstacle Channel')
#     plt.colorbar(im1, ax=axes[0])
    
#     # Plot free space TSDF
#     im2 = axes[1].imshow(tsdf_result[0, 1], cmap='RdYlBu_r', vmin=-3, vmax=3)
#     axes[1].set_title('Free Space Channel')
#     plt.colorbar(im2, ax=axes[1])
    
#     # Plot combined with LiDAR points
#     combined = tsdf_result[0, 0] - tsdf_result[0, 1]  # Obstacle - Free space
#     im3 = axes[2].imshow(combined, cmap='RdYlBu_r')
#     axes[2].set_title('Combined (Obs - Free)')
    
#     # Overlay LiDAR points
#     if lidar_data.ndim == 1:
#         lidar_data = lidar_data[None, :]
    
#     for i in range(360):
#         if lidar_data[0, i] > 0 and lidar_data[0, i] < max_range:
#             distance = lidar_data[0, i]
#             angle = angles[i]
#             x_obs = distance * np.cos(angle)
#             y_obs = distance * np.sin(angle)
#             x_obs_grid = center + x_obs * scale
#             y_obs_grid = center + y_obs * scale
            
#             if 0 <= x_obs_grid < img_dim and 0 <= y_obs_grid < img_dim:
#                 axes[2].plot(x_obs_grid, y_obs_grid, 'r.', markersize=2)
#     # Print 360 angle around the robot on the combined plot
#     for i in range(0, 360, 45):
#         angle_rad = np.deg2rad(i)
#         x_text = center + (max_range * 0.8) * np.cos(angle_rad) * scale
#         y_text = center + (max_range * 0.8) * np.sin(angle_rad) * scale
#         axes[2].text(x_text, y_text, f'{i}°', fontsize=8, ha='center', va='center')

#     # Mark robot position
#     axes[2].plot(center, center, 'go', markersize=8)
#     plt.colorbar(im3, ax=axes[2])
    
#     plt.tight_layout()
#     plt.show()
    
#     return fig

def visualize_tsdf_with_lidar(lidar_data, tsdf_result, img_dim=32, max_range=125.0):
    """
    Create a visualization showing LiDAR points overlaid on TSDF, including unknown regions.
    """
    center = img_dim // 2
    scale = img_dim / (2 * max_range)
    angles = np.deg2rad(np.arange(360))
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Plot obstacle TSDF
    im1 = axes[0].imshow(tsdf_result[0, 0], cmap='RdYlBu_r', vmin=-3, vmax=3)
    axes[0].set_title('Obstacle Channel')
    plt.colorbar(im1, ax=axes[0])
    
    # Plot free space TSDF
    im2 = axes[1].imshow(tsdf_result[0, 1], cmap='RdYlBu_r', vmin=-3, vmax=3)
    axes[1].set_title('Free Space Channel')
    plt.colorbar(im2, ax=axes[1])
    
    # Plot unknown mask
    im3 = axes[2].imshow(tsdf_result[0, 2], cmap='gray', vmin=0, vmax=1)
    axes[2].set_title('Unknown Mask (1 = unexplored)')
    plt.colorbar(im3, ax=axes[2])
    
    # Plot combined (Obstacle - Free) with unknown masked as transparent
    combined = tsdf_result[0, 0] - tsdf_result[0, 1]
    combined = np.clip(combined, -3, 3)  # Clamp before colormap

    unknown_mask = tsdf_result[0, 2]
    
    # Create RGBA image: lower alpha where unknown=1 (unexplored)
    cmap = plt.cm.RdYlBu_r
    normed_combined = (combined + 3) / 6  # Normalize to [0,1] for colormap
    colored = cmap(normed_combined)
    colored[..., -1] = 1.0 - unknown_mask  # Alpha channel
    
    axes[3].imshow(colored)
    axes[3].set_title('Combined (Obs - Free, Unknown Masked)')
    
    # Overlay LiDAR points
    if lidar_data.ndim == 1:
        lidar_data = lidar_data[None, :]
    
    for i in range(360):
        if lidar_data[0, i] > 0 and lidar_data[0, i] < max_range:
            distance = lidar_data[0, i]
            angle = angles[i]
            x_obs = distance * np.cos(angle)
            y_obs = distance * np.sin(angle)
            x_obs_grid = center + x_obs * scale
            y_obs_grid = center + y_obs * scale
            
            if 0 <= x_obs_grid < img_dim and 0 <= y_obs_grid < img_dim:
                axes[3].plot(x_obs_grid, y_obs_grid, 'r.', markersize=2)
    
    # Add angle labels (0–360°)
    for i in range(0, 360, 45):
        angle_rad = np.deg2rad(i)
        x_text = center + (max_range * 0.8) * np.cos(angle_rad) * scale
        y_text = center + (max_range * 0.8) * np.sin(angle_rad) * scale
        axes[3].text(x_text, y_text, f'{i}°', fontsize=8, ha='center', va='center')

    # Mark robot center
    axes[3].plot(center, center, 'go', markersize=8)

    # Add colorbar for combined image
    sm = plt.cm.ScalarMappable(cmap='RdYlBu_r', norm=plt.Normalize(vmin=-3, vmax=3))
    sm.set_array([])
    fig.colorbar(sm, ax=axes[3], fraction=0.046, pad=0.04, label='Combined TSDF Value')
    
    plt.tight_layout()
    plt.show()
    
    return fig


# Test and debug function
def test_tsdf_generation(lidar_data):
    """Test the TSDF generation and print diagnostics"""
    print(f"Input LiDAR shape: {lidar_data.shape}")
    print(f"LiDAR range: {lidar_data.min():.1f} to {lidar_data.max():.1f}")
    print(f"Valid readings: {np.sum((lidar_data > 0) & (lidar_data < 200))}/360")
    
    tsdf = lidar_to_tsdf_obstacles(lidar_data, img_dim=32, max_range=125.0, trunc_thresh=3)
    
    print(f"\nTSDF output shape: {tsdf.shape}")
    print(f"Obstacle channel - min: {tsdf[0,0].min():.2f}, max: {tsdf[0,0].max():.2f}, mean: {tsdf[0,0].mean():.2f}")
    print(f"Free space channel - min: {tsdf[0,1].min():.2f}, max: {tsdf[0,1].max():.2f}, mean: {tsdf[0,1].mean():.2f}")
    
    # Check center region (where robot is)
    center = 16
    print(f"\nCenter region (robot location):")
    print(f"Obstacle: {tsdf[0,0,center,center]:.2f}")
    print(f"Free space: {tsdf[0,1,center,center]:.2f}")
    
    return tsdf

sample_lidar2 = np.array([
    141., 140., 140., 140., 140., 140., 140., 141., 141., 141., 141., 142., 142., 142.,
    143., 143., 144., 144., 145., 146., 146., 147., 148., 149., 150., 151., 152., 153.,
    154., 155., 157., 158., 158., 153., 148., 144., 140., 136., 133., 129., 126., 124.,
    121., 118., 116., 114., 111., 109., 107., 106., 104., 102., 100., 99., 98., 96.,
    95., 94., 92., 91., 90., 89., 88., 87., 87., 86., 85., 84., 45., 42.,
    42., 41., 40., 40., 41., 40., 40., 40., 41., 41., 42., 41., 44., 44.,
    76., 76., 76., 76., 76., 76., 76., 75., 75., 75., 75., 75., 75., 76.,
    76., 76., 76., 76., 76., 76., 77., 77., 77., 77., 78., 78., 79., 79.,
    79., 80., 80., 81., 82., 82., 83., 83., 84., 85., 86., 87., 87., 88.,
    89., 90., 91., 92., 94., 95., 96., 97., 99., 100., 102., 104., 105., 107.,
    109., 111., 113., 116., 118., 121., 123., 126., 129., 133., 136., 140., 144., 148.,
    152., 152., 151., 149., 148., 147., 146., 145., 144., 143., 142., 141., 140., 139.,
    139., 138., 137., 137., 136., 136., 136., 135., 135., 135., 134., 134., 134., 134.,
    134., 134., 134., 134., 134., 134., 134., 134., 135., 135., 135., 136., 136., 136.,
    137., 137., 138., 139., 139., 140., 141., 142., 143., 144., 145., 146., 147., 148.,
    149., 151., 152., 153., 155., 157., 158., 160., 162., 164., 166., 168., 171., 173.,
    176., 178., 181., 184., 187., 190., 194., 198., 200., 200., 200., 200., 200., 200.,
    200., 200., 200., 200., 200., 200., 200., 200., 200., 200., 200., 200., 200., 200.,
    200., 200., 200., 200., 200., 200., 200., 200., 200., 200., 200., 200., 200., 200.,
    200., 200., 200., 200., 200., 200., 200., 200., 200., 200., 200., 200., 200., 200.,
    200., 200., 200., 200., 200., 200., 200., 200., 200., 200., 200., 200., 200., 200.,
    200., 44., 39., 39., 39., 37., 36., 35., 35., 35., 34., 33., 34., 33.,
    32., 33., 32., 32., 32., 32., 32., 32., 32., 32., 33., 33., 33., 33.,
    34., 33., 34., 34., 35., 36., 37., 37., 39., 40., 41., 46., 161., 160.,
    158., 157., 155., 154., 153., 152., 151., 150., 149., 148., 147., 146., 146., 145.,
    144., 144., 143., 143., 142., 142., 142., 141., 141., 141.
], dtype=np.float32)

sample_lidar3 = np.array([
    126., 126., 126., 126., 126., 126., 126., 127., 126., 126., 126., 126., 126., 126.,
    126., 126., 126., 126., 126., 126., 126., 126., 126., 126., 126., 126., 126., 126.,
    127., 126., 126., 126., 126., 126., 126., 126., 126., 126., 126., 126., 126., 126.,
    126., 126., 126., 126., 126., 127., 126.,  43.,  40.,  39.,  38.,  36.,  36.,  36.,
    35.,  35.,  34.,  34.,  33.,  33.,  33.,  33.,  32.,  33.,  33.,  32.,  32.,  33.,
    32.,  32.,  33.,  33.,  33.,  33.,  33.,  35.,  34.,  35.,  36.,  36.,  36.,  38.,
    40.,  40.,  44., 126., 126., 126., 126., 126., 127., 126., 126., 115., 113., 113.,
    113., 112., 114., 126., 126., 126., 126., 126., 126., 126., 126., 126., 126., 126.,
    126., 127., 126., 127., 126., 127., 126., 127., 126., 126., 126., 126., 126., 126.,
    126., 126., 126., 126., 126., 126., 126., 126., 126., 127., 126., 126., 126., 126.,
    126., 126., 126., 126., 126., 126., 126., 127., 126., 126., 126., 126., 126., 126.,
    126., 127., 126., 126., 126., 127., 126., 126., 126., 126., 126., 127., 127., 127.,
    127., 127., 126., 126., 126., 126., 127., 126., 126., 126., 126., 126., 126., 126.,
    126., 126., 126., 126., 126., 126., 126., 126., 126., 126., 126., 126., 126., 126.,
    126., 126., 126., 126., 126., 126., 126., 126., 126., 126., 126., 126., 126., 126.,
    126., 126., 126., 127., 126., 126., 126., 126., 127., 127., 127., 127., 127., 127.,
    126., 126., 126., 126., 126., 126., 126., 126., 127., 126., 126., 126., 126., 126.,
    126., 126., 127., 126., 127., 126., 126., 126., 126., 126., 126., 126., 126., 127.,
    126., 126., 126., 126., 126., 126., 126., 126., 126., 127., 126., 126., 126., 126.,
    126., 126., 127., 126., 126., 126., 126., 126., 126., 126., 126., 126., 126., 126.,
    126., 126., 126., 126., 126., 126., 126., 126., 126., 126., 126., 126., 126., 126.,
    126., 126., 126., 126., 126., 126., 126., 126., 126., 126., 126., 127., 126., 126.,
    127., 127., 128., 128., 127., 126., 130., 126., 126., 126., 126., 126., 127., 126.,
    126., 127., 126., 126., 126., 126., 126., 127., 126., 126., 126., 126., 126., 126.,
    126., 126., 126., 127., 126., 126., 126., 126., 126., 126., 126., 126., 126., 126.,
    126., 126., 126., 126., 126., 126., 127., 126., 126., 126.
], dtype=np.float32)

tsdf = test_tsdf_generation(sample_lidar4)
visualize_tsdf_with_lidar(sample_lidar4, tsdf, img_dim=32)

def meter2pixel(x_in_m, m2p_ratio):
    """ convert world meter into pixel"""
    return np.round(x_in_m*m2p_ratio).astype(int)

print(meter2pixel(5, 25)) # laser range in pixel
