#!/home/ckelley/bundle-adjustment-compare/venv/bin/python3
"""
Enhanced 3D visualization of bundle adjustment results for presentation.
Shows camera trajectory and 3D point cloud with multiple views and professional styling.
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyBboxPatch
import os

# Detect project root
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
output_dir = os.path.join(project_root, 'output')

def quaternion_to_rotation_matrix(q):
    """Convert quaternion [qw, qx, qy, qz] to 3x3 rotation matrix"""
    qw, qx, qy, qz = q / np.linalg.norm(q)  # Normalize
    return np.array([
        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
        [2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)],
        [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2)]
    ])

def plot_camera_frustum(ax, position, quaternion, scale=2.0, color='red', alpha=0.3):
    """Plot camera frustum showing field of view"""
    # Get rotation matrix from quaternion
    R = quaternion_to_rotation_matrix(quaternion)
    
    # Define frustum corners in camera frame (simple pyramid)
    # Front face corners (at focal length distance)
    frustum_points = np.array([
        [0, 0, 0],           # Camera center
        [-0.5, -0.5, 1],     # Bottom-left
        [0.5, -0.5, 1],      # Bottom-right
        [0.5, 0.5, 1],       # Top-right
        [-0.5, 0.5, 1],      # Top-left
    ]) * scale
    
    # Transform to world frame
    world_points = position + (R @ frustum_points.T).T
    
    # Draw frustum edges
    edges = [
        (0, 1), (0, 2), (0, 3), (0, 4),  # From center to corners
        (1, 2), (2, 3), (3, 4), (4, 1)   # Rectangle at front
    ]
    
    for start, end in edges:
        ax.plot3D(*zip(world_points[start], world_points[end]), 
                 color=color, linewidth=1, alpha=alpha)

def create_enhanced_visualization():
    """Create multi-panel enhanced visualization"""
    
    # Load optimized poses if available, otherwise use initial
    if os.path.exists(os.path.join(output_dir, 'dogleg_poses_optimized.txt')):
        poses_file = 'dogleg_poses_optimized.txt'
        points_file = 'dogleg_points_optimized.txt'
        title_suffix = '(Optimized - Dogleg Solver)'
    else:
        poses_file = 'ba_robotcar_poses.txt'
        points_file = 'ba_robotcar_points.txt'
        title_suffix = '(Initial Estimate)'
    
    # Load poses: id tx ty tz qx qy qz qw (optimized) or id tx ty tz qw qx qy qz (initial)
    poses = np.loadtxt(os.path.join(output_dir, poses_file))
    frame_ids = poses[:, 0]
    translations = poses[:, 1:4]  # tx ty tz
    
    # Handle quaternion order difference
    if 'optimized' in poses_file:
        # Optimized format: id tx ty tz qx qy qz qw
        quat_xyzw = poses[:, 4:8]
        quaternions = np.column_stack([quat_xyzw[:, 3], quat_xyzw[:, 0:3]])  # → qw qx qy qz
    else:
        # Initial format: id tx ty tz qw qx qy qz
        quaternions = poses[:, 4:8]
    
    print(f"Loaded {len(poses)} poses from {poses_file}")
    
    # Load points: id X Y Z
    points_data = np.loadtxt(os.path.join(output_dir, points_file))
    points = points_data[:, 1:4]  # X Y Z
    
    # Filter outliers
    distances = np.linalg.norm(points, axis=1)
    threshold = np.percentile(distances, 99)  # Keep 99% of points
    points = points[distances < threshold]
    print(f"Filtered to {len(points)} points (removed outliers >{threshold:.1f}m)")
    
    # Calculate trajectory statistics
    trajectory_length = np.sum(np.linalg.norm(np.diff(translations, axis=0), axis=1))
    extent = translations.max(axis=0) - translations.min(axis=0)
    
    # Create figure with better layout
    fig = plt.figure(figsize=(22, 12))
    fig.suptitle(f'Bundle Adjustment 3D Reconstruction - All 50 Frames {title_suffix}', 
                 fontsize=18, fontweight='bold', y=0.96)
    
    # Color map for trajectory (time progression)
    colors = plt.cm.viridis(np.linspace(0, 1, len(translations)))
    
    # ==== Subplot 1: Main 3D view (larger, spans 2 rows) ====
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    
    # Plot point cloud with density coloring
    point_colors = points[:, 2]  # Color by height (Z)
    scatter = ax1.scatter(points[:, 0], points[:, 1], points[:, 2], 
                         c=point_colors, cmap='gray', marker='.', 
                         s=0.5, alpha=0.3, label='3D Landmarks')
    
    # Plot trajectory with color gradient
    for i in range(len(translations) - 1):
        ax1.plot3D(translations[i:i+2, 0], 
                  translations[i:i+2, 1], 
                  translations[i:i+2, 2], 
                  color=colors[i], linewidth=3, alpha=0.8)
    
    # Plot camera positions
    ax1.scatter(translations[:, 0], translations[:, 1], translations[:, 2], 
               c=colors, marker='o', s=100, edgecolors='black', linewidths=1.5,
               label='Camera Poses')
    
    # Add camera frustums at regular intervals (every 5 frames)
    for i in range(0, len(translations), 5):
        plot_camera_frustum(ax1, translations[i], quaternions[i], 
                          scale=5.0, color=colors[i], alpha=0.4)
    
    # Mark start and end with offset for visibility
    ax1.scatter(*translations[0], color='lime', marker='*', s=800, 
               edgecolors='black', linewidths=3, label='Start (Frame 0)', zorder=100)
    ax1.scatter(*translations[-1], color='red', marker='*', s=800, 
               edgecolors='black', linewidths=3, label='End (Frame 49)', zorder=100)
    
    # Add frame number annotations at key points
    for i in [0, 12, 24, 37, 49]:
        ax1.text(translations[i, 0], translations[i, 1], translations[i, 2] + 2,
                f'{i}', fontsize=10, fontweight='bold', 
                bbox=dict(boxstyle='round,pad=0.3', facecolor=colors[i], alpha=0.7),
                ha='center')
    
    ax1.set_xlabel('X (meters)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Y (meters)', fontsize=12, fontweight='bold')
    ax1.set_zlabel('Z (meters)', fontsize=12, fontweight='bold')
    ax1.set_title('3D Trajectory & Point Cloud (50 Frames)', fontsize=14, pad=15)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.view_init(elev=25, azim=135)
    ax1.grid(True, alpha=0.3)
    
    # ==== Right panel: Multiple temporal analysis plots ====
    # Create GridSpec for right side
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(3, 2, figure=fig, left=0.55, right=0.98, 
                  top=0.90, bottom=0.08, hspace=0.35, wspace=0.3)
    
    # Plot 1: Camera position over time
    ax2 = fig.add_subplot(gs[0, :])
    frame_indices = np.arange(len(translations))
    ax2.plot(frame_indices, translations[:, 0], 'o-', color='#1f77b4', 
            linewidth=2, markersize=4, label='X position', alpha=0.8)
    ax2.plot(frame_indices, translations[:, 1], 's-', color='#ff7f0e', 
            linewidth=2, markersize=4, label='Y position', alpha=0.8)
    ax2.plot(frame_indices, translations[:, 2], '^-', color='#2ca02c', 
            linewidth=2, markersize=4, label='Z position', alpha=0.8)
    ax2.set_xlabel('Frame Number', fontsize=10, fontweight='bold')
    ax2.set_ylabel('Position (meters)', fontsize=10, fontweight='bold')
    ax2.set_title('Camera Position Evolution Across All 50 Frames', fontsize=12, pad=10)
    ax2.legend(loc='best', fontsize=9, ncol=3)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-1, len(translations))
    
    # Plot 2: Inter-frame movement (velocity proxy)
    ax3 = fig.add_subplot(gs[1, :])
    movements = np.linalg.norm(np.diff(translations, axis=0), axis=1)
    ax3.plot(frame_indices[:-1], movements, 'o-', color='purple', 
            linewidth=2, markersize=5, alpha=0.7)
    ax3.fill_between(frame_indices[:-1], movements, alpha=0.3, color='purple')
    ax3.set_xlabel('Frame Number', fontsize=10, fontweight='bold')
    ax3.set_ylabel('Movement (meters)', fontsize=10, fontweight='bold')
    ax3.set_title('Inter-Frame Camera Movement', fontsize=12, pad=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(-1, len(translations)-1)
    
    # Add statistics
    avg_movement = np.mean(movements)
    ax3.axhline(avg_movement, color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {avg_movement:.2f}m', alpha=0.7)
    ax3.legend(loc='best', fontsize=9)
    
    # Plot 3: Statistics panel
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('off')
    
    # Create comprehensive text summary
    stats_text = f"""RECONSTRUCTION STATISTICS (50 Frames)

Problem Size:          Trajectory Analysis:
• Cameras: {len(translations)}              • Total length: {trajectory_length:.1f} m
• 3D Landmarks: {len(points):,}        • Avg movement: {avg_movement:.2f} m/frame
• Total observations: ~{len(points)*3}    • Max movement: {np.max(movements):.2f} m

Spatial Extent:        Camera Path:
• X: {extent[0]:.1f} m              • Start pos: ({translations[0, 0]:.1f}, {translations[0, 1]:.1f}, {translations[0, 2]:.1f})
• Y: {extent[1]:.1f} m              • End pos: ({translations[-1, 0]:.1f}, {translations[-1, 1]:.1f}, {translations[-1, 2]:.1f})
• Z: {extent[2]:.1f} m              • Net displacement: {np.linalg.norm(translations[-1] - translations[0]):.1f} m

Color Encoding: Viridis colormap represents temporal progression (Frame 0→49)
Camera frustums indicate viewing directions at 5-frame intervals"""
    
    ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes,
            fontsize=10, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='#e8f4f8', alpha=0.9, edgecolor='#666', linewidth=2))
    
    # Adjust layout manually instead of tight_layout (conflicts with GridSpec)
    plt.subplots_adjust(left=0.05, right=0.98, top=0.93, bottom=0.05)
    
    # Save high-resolution version
    output_path = os.path.join(output_dir, 'robotcar_3d_map_enhanced.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f'\nEnhanced visualization saved to {output_path}')
    
    return output_path

if __name__ == '__main__':
    create_enhanced_visualization()
