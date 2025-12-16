#!/home/ckelley/bundle-adjustment-compare/venv/bin/python3
import numpy as np
import json
import os
from PIL import Image
import matplotlib.pyplot as plt

# Paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
output_dir = os.path.join(project_root, 'output')
sample_dir = os.path.join(project_root, 'sample')
data_dir = os.path.join(sample_dir, 'stereo', 'left')

# Load timestamps to map frame IDs to image filenames
timestamps = np.loadtxt(os.path.join(sample_dir, 'stereo.timestamps'), dtype=str)
frame_to_timestamp = {i: ts[0] for i, ts in enumerate(timestamps)}

# Load intrinsics
with open(os.path.join(output_dir, 'ba_intrinsics_stereo_left.json')) as f:
    intr = json.load(f)
fx, fy, cx, cy = intr['fx'], intr['fy'], intr['cx'], intr['cy']

# Load OPTIMIZED poses and points (output from BA, not initial guesses)
# File format: id tx ty tz qx qy qz qw
poses = np.loadtxt(os.path.join(output_dir, 'dogleg_poses_optimized.txt'))
translations = poses[:, 1:4]  # tx ty tz
quat_xyzw = poses[:, 4:8]  # qx qy qz qw
# Convert to [qw, qx, qy, qz] for our rotation function
quaternions = np.column_stack([quat_xyzw[:, 3], quat_xyzw[:, 0], quat_xyzw[:, 1], quat_xyzw[:, 2]])

points = np.loadtxt(os.path.join(output_dir, 'dogleg_points_optimized.txt'))[:, 1:4]

# Load observations with original pixel locations
# Format: frame_id point_id u_observed v_observed
obs = np.loadtxt(os.path.join(output_dir, 'ba_robotcar_observations.txt'))

# Build per-frame observation data: {frame_id: [(point_id, u_obs, v_obs), ...]}
frame_obs = {}
for row in obs:
    frame_id = int(row[0])
    point_id = int(row[1])
    u_obs = row[2]
    v_obs = row[3]
    
    if frame_id not in frame_obs:
        frame_obs[frame_id] = []
    frame_obs[frame_id].append((point_id, u_obs, v_obs))

# Function to apply quaternion rotation
def quat_rotate(q, v):
    """Rotate vector v by quaternion q = [qw, qx, qy, qz]"""
    qw, qx, qy, qz = q
    # Convert to rotation matrix (from quaternion)
    R = np.array([
        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
        [2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)],
        [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2)]
    ])
    return R @ v

# Function to project 3D point to 2D using pose
def project_point(point, trans, quat):
    """Project 3D world point to 2D image coordinates
    
    BA module uses: p_cam = R * p_world + t
    So pose quaternion is world-to-camera rotation, t is world-to-camera translation
    """
    # Normalize quaternion
    q = quat / np.linalg.norm(quat)
    
    # Transform to camera frame: p_cam = R * p_world + t (same as BA module)
    pc = quat_rotate(q, point)  # Rotate
    pc = pc + trans  # Translate
    
    # Check if point is in front of camera
    if pc[2] <= 0:
        return None, None
    
    # Project to image plane
    xp = pc[0] / pc[2]
    yp = pc[1] / pc[2]
    
    # Apply intrinsics
    u = fx * xp + cx
    v = fy * yp + cy
    
    return u, v

# Select frames to visualize - all 50 frames
frames_to_viz = list(range(50))

for frame_id in frames_to_viz:
    # Use timestamp for image filename
    if frame_id not in frame_to_timestamp:
        print(f"Frame {frame_id} has no corresponding timestamp - skipping")
        continue
    
    timestamp = frame_to_timestamp[frame_id]
    img_path = os.path.join(data_dir, f"{timestamp}.png")
    
    if not os.path.exists(img_path):
        print(f"Image {img_path} not found - skip or adjust path")
        continue
    img = np.array(Image.open(img_path))

    trans = translations[frame_id]
    quat = quaternions[frame_id]

    # Get observations for this frame
    if frame_id not in frame_obs:
        print(f"No observations for frame {frame_id} - skipping")
        continue
    
    observations = frame_obs[frame_id]  # List of (point_id, u_obs, v_obs)
    
    # Compute reprojections and errors
    reprojection_data = []
    errors = []

    for point_id, u_obs, v_obs in observations:
        # Project the optimized 3D point
        pt = points[point_id]
        u_pred, v_pred = project_point(pt, trans, quat)
        
        # Check if projection is valid
        if u_pred is not None and v_pred is not None:
            # Check if both observed and predicted are within image bounds
            if (0 <= u_obs < img.shape[1] and 0 <= v_obs < img.shape[0] and
                0 <= u_pred < img.shape[1] and 0 <= v_pred < img.shape[0]):
                
                error = np.sqrt((u_pred - u_obs)**2 + (v_pred - v_obs)**2)
                reprojection_data.append((u_obs, v_obs, u_pred, v_pred, error))
                errors.append(error)
    
    if not reprojection_data:
        print(f"No valid reprojections for frame {frame_id} - skipping")
        continue
    
    # Calculate statistics
    errors = np.array(errors)
    mean_error = np.mean(errors)
    max_error = np.max(errors)

    # Plot image with overlays
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.imshow(img, cmap='gray')
    
    # Draw error vectors with color coding
    for u_obs, v_obs, u_pred, v_pred, error in reprojection_data:
        # Color by error magnitude (blue=good, red=bad)
        color = plt.cm.RdYlGn_r(error / max(5.0, max_error))  # Scale to 5px max
        
        # Draw line from observed to predicted
        ax.plot([u_obs, u_pred], [v_obs, v_pred], 
                color=color, linewidth=1, alpha=0.6)
        
        # Draw observed point (red circle)
        ax.scatter(u_obs, v_obs, c='red', s=30, marker='o', 
                  edgecolors='white', linewidths=0.5, alpha=0.7)
        
        # Draw predicted point (green X)
        ax.scatter(u_pred, v_pred, c='lime', s=30, marker='x', 
                  linewidths=1.5, alpha=0.8)
    
    # Create legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
               markersize=8, label='Observed (before BA)'),
        Line2D([0], [0], marker='x', color='lime', markersize=8, 
               linewidth=2, label='Predicted (after BA)'),
        Line2D([0], [0], color='red', linewidth=2, 
               label=f'Error (mean: {mean_error:.2f}px)')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=11, framealpha=0.9)
    
    ax.set_title(f'Frame {frame_id}: Reprojection Error Visualization\n'
                f'{len(reprojection_data)} points | Mean error: {mean_error:.2f}px | Max: {max_error:.2f}px',
                fontsize=14, pad=10)
    ax.axis('off')
    
    # Save with zero-padded frame number for proper video sorting
    out_path = os.path.join(output_dir, 'reproj_frames', f'frame_{frame_id:04d}.png')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Frame {frame_id}: {mean_error:.2f}px ({len(reprojection_data)} points)')
