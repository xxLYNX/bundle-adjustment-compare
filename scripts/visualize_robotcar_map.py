#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import sys

# Detect project root
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
output_dir = os.path.join(project_root, 'output')

# Load poses from ba_robotcar_poses.txt
# Format: id tx ty tz qw qx qy qz
poses = np.loadtxt(os.path.join(output_dir, 'ba_robotcar_poses.txt'))
frame_ids = poses[:, 0]
translations = poses[:, 1:4]  # tx ty tz
quaternions = poses[:, 4:]   # qw qx qy qz (not used for basic plot, but could add camera orientations)
print(f"Loaded {len(poses)} poses")

# Load points from ba_robotcar_points.txt
# Format: id X Y Z
points_data = np.loadtxt(os.path.join(output_dir, 'ba_robotcar_points.txt'))
point_ids = points_data[:, 0]
points = points_data[:, 1:4]  # X Y Z

# Optional: Filter outlier points (e.g., remove extremely far points for cleaner viz)
# Adjust thresholds based on your data (e.g., RobotCar points can have far-off noise)
points = points[np.linalg.norm(points, axis=1) < 1000]  # Example: remove points >1000 units away
print(f"Filtered to {len(points)} points (removed outliers >1000m)")
# Create 3D figure
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot 3D points as blue scatter
ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b', marker='.', s=1, alpha=0.5, label='3D Landmarks')

# Plot camera poses as red points
ax.scatter(translations[:, 0], translations[:, 1], translations[:, 2], c='r', marker='o', s=50, label='Camera Poses')

# Add lines connecting consecutive poses to show trajectory
for i in range(len(translations) - 1):
    ax.plot(translations[i:i+2, 0], translations[i:i+2, 1], translations[i:i+2, 2], 'r-', linewidth=2)

# Set labels, title, and view
ax.set_xlabel('X (meters)')
ax.set_ylabel('Y (meters)')
ax.set_zlabel('Z (meters)')
ax.set_title('Sparse 3D Map from Bundle Adjustment (Oxford RobotCar Subset)')
ax.legend()
ax.view_init(elev=20, azim=120)  # Adjust viewing angle for better perspective

# Save and show the plot
output_path = os.path.join(output_dir, 'robotcar_3d_map.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
# plt.show()  # Uncomment to display interactively
print(f'Visualization saved to {output_path}')
