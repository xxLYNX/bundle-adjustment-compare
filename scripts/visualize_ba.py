#!/usr/bin/env python3
"""
Visualize Bundle Adjustment results from the output directory.
Creates plots comparing camera poses, 3D points, and solver performance.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
import os
import re


def load_poses(filepath):
    """Load camera poses from file."""
    poses = []
    with open(filepath, 'r') as f:
        for line in f:
            vals = list(map(float, line.strip().split()))
            if len(vals) == 7:  # qw qx qy qz tx ty tz
                poses.append(vals)
    return np.array(poses)


def load_points(filepath):
    """Load 3D points from file."""
    points = []
    with open(filepath, 'r') as f:
        for line in f:
            vals = list(map(float, line.strip().split()))
            if len(vals) == 3:  # x y z
                points.append(vals)
    return np.array(points)


def load_observations(filepath):
    """Load observations from file."""
    observations = []
    with open(filepath, 'r') as f:
        for line in f:
            vals = list(map(int, line.strip().split()[:2]))  # frame_id point_id
            if len(vals) == 2:
                observations.append(vals)
    return np.array(observations)


def parse_ba_analysis(filepath):
    """Parse BA analysis file to extract metrics."""
    metrics = {}
    with open(filepath, 'r') as f:
        content = f.read()
        
        # Extract solver type
        match = re.search(r'Solver: (.+)', content)
        if match:
            metrics['solver'] = match.group(1)
        
        # Extract timing
        match = re.search(r'BA Optimization Time: ([\d.]+) seconds', content)
        if match:
            metrics['ba_time'] = float(match.group(1))
        
        # Extract iterations
        match = re.search(r'Total Iterations: (\d+)', content)
        if match:
            metrics['total_iterations'] = int(match.group(1))
            
        match = re.search(r'Successful Steps: (\d+)', content)
        if match:
            metrics['successful_steps'] = int(match.group(1))
        
        # Extract costs
        match = re.search(r'Initial Cost: ([\d.e+-]+)', content)
        if match:
            metrics['initial_cost'] = float(match.group(1))
            
        match = re.search(r'Final Cost: ([\d.e+-]+)', content)
        if match:
            metrics['final_cost'] = float(match.group(1))
        
        # Extract RMS error
        match = re.search(r'RMS Reprojection Error: ([\d.]+) pixels', content)
        if match:
            metrics['rms_error'] = float(match.group(1))
        
        # Extract time per iteration
        match = re.search(r'Time per Iteration: ([\d.]+) ms', content)
        if match:
            metrics['time_per_iter'] = float(match.group(1))
    
    return metrics


def visualize_trajectory(poses, title="Camera Trajectory"):
    """Plot 3D camera trajectory."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract positions
    positions = poses[:, 4:]  # tx, ty, tz
    
    # Plot trajectory
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
            'b-', linewidth=2, label='Trajectory')
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
               c='r', marker='o', s=100, label='Camera Poses')
    
    # Mark start and end
    ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2],
               c='g', marker='*', s=300, label='Start')
    ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2],
               c='orange', marker='s', s=200, label='End')
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    
    return fig


def visualize_point_cloud(points, title="3D Point Cloud", max_points=5000):
    """Plot 3D point cloud."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Subsample if too many points
    if len(points) > max_points:
        indices = np.random.choice(len(points), max_points, replace=False)
        points = points[indices]
    
    # Color by height (z coordinate)
    colors = points[:, 2]
    
    scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                        c=colors, cmap='viridis', s=1, alpha=0.5)
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(title)
    plt.colorbar(scatter, label='Height (m)', shrink=0.5)
    ax.grid(True)
    
    return fig


def visualize_scene(poses, points, title="Bundle Adjustment Scene", max_points=5000):
    """Plot both trajectory and point cloud together."""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Subsample points if needed
    if len(points) > max_points:
        indices = np.random.choice(len(points), max_points, replace=False)
        points_sub = points[indices]
    else:
        points_sub = points
    
    # Plot point cloud
    ax.scatter(points_sub[:, 0], points_sub[:, 1], points_sub[:, 2],
              c='lightgray', s=1, alpha=0.3, label='3D Points')
    
    # Plot trajectory
    positions = poses[:, 4:]
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2],
           'r-', linewidth=3, label='Camera Trajectory')
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
              c='red', marker='o', s=100, edgecolors='k', linewidths=2)
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    
    return fig


def compare_solvers(lm_metrics, dogleg_metrics, benchmark_data=None):
    """Create comparison plots for LM vs Dogleg."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Levenberg-Marquardt vs Dogleg Comparison', fontsize=16)
    
    solvers = ['LM', 'Dogleg']
    
    # Extract error bars if benchmark data available
    lm_time_err = None
    dog_time_err = None
    if benchmark_data and 'results' in benchmark_data:
        if 'lm' in benchmark_data['results'] and 'ba_time' in benchmark_data['results']['lm']:
            lm_time_err = benchmark_data['results']['lm']['ba_time'].get('std', 0)
        if 'dogleg' in benchmark_data['results'] and 'ba_time' in benchmark_data['results']['dogleg']:
            dog_time_err = benchmark_data['results']['dogleg']['ba_time'].get('std', 0)
    
    # Plot 1: Total Time
    ax = axes[0, 0]
    times = [lm_metrics.get('ba_time', 0), dogleg_metrics.get('ba_time', 0)]
    errors = [lm_time_err, dog_time_err] if lm_time_err and dog_time_err else None
    
    bars = ax.bar(solvers, times, color=['#2E86AB', '#A23B72'], yerr=errors, capsize=5)
    ax.set_ylabel('Time (seconds)')
    title = 'Total Optimization Time'
    if errors:
        title += f' (n={benchmark_data.get("num_runs", "?")})'
    ax.set_title(title)
    ax.grid(axis='y', alpha=0.3)
    for bar, time in zip(bars, times):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{time:.2f}s', ha='center', va='bottom')
    
    # Plot 2: Iterations
    ax = axes[0, 1]
    iters = [lm_metrics.get('total_iterations', 0), 
             dogleg_metrics.get('total_iterations', 0)]
    
    # Extract iteration error bars if available
    iter_errors = [0, 0]
    if benchmark_data and 'results' in benchmark_data:
        if 'lm' in benchmark_data['results'] and 'total_iterations' in benchmark_data['results']['lm']:
            iter_errors[0] = benchmark_data['results']['lm']['total_iterations'].get('std', 0)
        if 'dogleg' in benchmark_data['results'] and 'total_iterations' in benchmark_data['results']['dogleg']:
            iter_errors[1] = benchmark_data['results']['dogleg']['total_iterations'].get('std', 0)
    
    bars = ax.bar(solvers, iters, yerr=iter_errors, capsize=5, color=['#2E86AB', '#A23B72'])
    ax.set_ylabel('Iterations')
    title = 'Total Iterations'
    if benchmark_data:
        title += f' (n={benchmark_data.get("num_runs", "?")})'
    ax.set_title(title)
    ax.grid(axis='y', alpha=0.3)
    for bar, it in zip(bars, iters):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{it}', ha='center', va='bottom')
    
    # Plot 3: Final Cost
    ax = axes[1, 0]
    costs = [lm_metrics.get('final_cost', 0), dogleg_metrics.get('final_cost', 0)]
    
    # Extract cost error bars if available
    cost_errors = [0, 0]
    if benchmark_data and 'results' in benchmark_data:
        if 'lm' in benchmark_data['results'] and 'final_cost' in benchmark_data['results']['lm']:
            cost_errors[0] = benchmark_data['results']['lm']['final_cost'].get('std', 0)
        if 'dogleg' in benchmark_data['results'] and 'final_cost' in benchmark_data['results']['dogleg']:
            cost_errors[1] = benchmark_data['results']['dogleg']['final_cost'].get('std', 0)
    
    bars = ax.bar(solvers, costs, yerr=cost_errors, capsize=5, color=['#2E86AB', '#A23B72'])
    ax.set_ylabel('Cost')
    title = 'Final Cost'
    if benchmark_data:
        title += f' (n={benchmark_data.get("num_runs", "?")})'
    ax.set_title(title)
    ax.set_yscale('log')
    ax.grid(axis='y', alpha=0.3)
    for bar, cost in zip(bars, costs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{cost:.2e}', ha='center', va='bottom', fontsize=9)
    
    # Plot 4: RMS Error
    ax = axes[1, 1]
    errors = [lm_metrics.get('rms_error', 0), dogleg_metrics.get('rms_error', 0)]
    
    # Extract RMS error bars if available
    rms_errors = [0, 0]
    if benchmark_data and 'results' in benchmark_data:
        if 'lm' in benchmark_data['results'] and 'rms_error' in benchmark_data['results']['lm']:
            rms_errors[0] = benchmark_data['results']['lm']['rms_error'].get('std', 0)
        if 'dogleg' in benchmark_data['results'] and 'rms_error' in benchmark_data['results']['dogleg']:
            rms_errors[1] = benchmark_data['results']['dogleg']['rms_error'].get('std', 0)
    
    bars = ax.bar(solvers, errors, yerr=rms_errors, capsize=5, color=['#2E86AB', '#A23B72'])
    ax.set_ylabel('RMS Error (pixels)')
    title = 'RMS Reprojection Error'
    if benchmark_data:
        title += f' (n={benchmark_data.get("num_runs", "?")})'
    ax.set_title(title)
    ax.grid(axis='y', alpha=0.3)
    for bar, err in zip(bars, errors):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{err:.3f}px', ha='center', va='bottom')
    
    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description='Visualize Bundle Adjustment results')
    parser.add_argument('--output_dir', type=str, default='output',
                       help='Directory containing BA output files')
    parser.add_argument('--show', action='store_true',
                       help='Show plots interactively')
    parser.add_argument('--save', action='store_true', default=True,
                       help='Save plots to output directory')
    args = parser.parse_args()
    
    output_dir = args.output_dir
    
    # Load data
    poses_file = os.path.join(output_dir, 'ba_robotcar_poses.txt')
    points_file = os.path.join(output_dir, 'ba_robotcar_points.txt')
    
    if not os.path.exists(poses_file) or not os.path.exists(points_file):
        print(f"Error: Required files not found in {output_dir}")
        return
    
    print(f"Loading data from {output_dir}...")
    poses = load_poses(poses_file)
    points = load_points(points_file)
    print(f"Loaded {len(poses)} poses and {len(points)} points")
    
    # Create visualizations
    figures = []
    
    # 1. Combined scene view
    print("Creating scene visualization...")
    fig = visualize_scene(poses, points, "Bundle Adjustment: Trajectory and Points")
    figures.append(('scene', fig))
    
    # 2. Trajectory only
    print("Creating trajectory visualization...")
    fig = visualize_trajectory(poses, "Camera Trajectory")
    figures.append(('trajectory', fig))
    
    # 3. Point cloud only
    print("Creating point cloud visualization...")
    fig = visualize_point_cloud(points, "3D Point Cloud from Triangulation")
    figures.append(('pointcloud', fig))
    
    # 4. Solver comparison if both analyses exist
    lm_file = os.path.join(output_dir, 'lm_BA.txt')
    dogleg_file = os.path.join(output_dir, 'dogleg_BA.txt')
    
    # Load benchmark data if available
    benchmark_file = os.path.join(output_dir, 'benchmark_results.json')
    benchmark_data = None
    if os.path.exists(benchmark_file):
        try:
            with open(benchmark_file, 'r') as f:
                benchmark_data = json.load(f)
            print(f"Loaded benchmark data (n={benchmark_data.get('num_runs', '?')} runs)")
        except Exception as e:
            print(f"Warning: Could not load benchmark data: {e}")
    
    if os.path.exists(lm_file) and os.path.exists(dogleg_file):
        print("Creating solver comparison...")
        lm_metrics = parse_ba_analysis(lm_file)
        dogleg_metrics = parse_ba_analysis(dogleg_file)
        fig = compare_solvers(lm_metrics, dogleg_metrics, benchmark_data)
        figures.append(('comparison', fig))
    
    # Save figures
    if args.save:
        print("\nSaving figures...")
        for name, fig in figures:
            filename = os.path.join(output_dir, f'{name}.png')
            fig.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"  Saved: {filename}")
    
    # Show figures
    if args.show:
        plt.show()
    
    print("\nVisualization complete!")


if __name__ == '__main__':
    main()
