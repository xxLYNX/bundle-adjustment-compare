#!/usr/bin/env python3
import os
import sys
import argparse
import numpy as np

# Ensure SDK python modules are on sys.path
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SDK_PY = os.path.join(os.path.dirname(THIS_DIR), "robotcar-dataset-sdk-3.1", "python")
if SDK_PY not in sys.path:
    sys.path.insert(0, SDK_PY)

from build_pointcloud import build_pointcloud  # SDK function


def main():
    parser = argparse.ArgumentParser(
        description="Build a pointcloud from RobotCar lidar + poses (no Open3D)."
    )
    parser.add_argument(
        "--dataset_root",
        required=True,
        help="Path to traversal root (where lms_front, gps/ins.csv, etc. live).",
    )
    parser.add_argument(
        "--lidar",
        default="lms_front",
        help="Lidar folder name (e.g. lms_front, lms_rear, ldmrs, velodyne_left, velodyne_right).",
    )
    parser.add_argument(
        "--poses_file",
        default="gps/ins.csv",
        help="Poses CSV filename relative to dataset_root (e.g. gps/ins.csv, vo/vo.csv).",
    )
    parser.add_argument(
        "--extrinsics_dir",
        required=True,
        help="Path to the SDK 'extrinsics' directory.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Time window length in seconds for the pointcloud (starting from first lidar timestamp).",
    )
    parser.add_argument(
        "--output",
        default="pointcloud.npz",
        help="Output file (.npz) to save points and reflectance.",
    )
    args = parser.parse_args()

    # Paths
    lidar_dir = os.path.join(args.dataset_root, args.lidar)
    timestamps_path = os.path.join(args.dataset_root, args.lidar + ".timestamps")
    poses_path = os.path.join(args.dataset_root, args.poses_file)

    # Read first lidar timestamp to define a window
    with open(timestamps_path) as f:
        first_line = next(f)
    start_time = int(first_line.split()[0])
    end_time = start_time + int(args.duration * 1e6)  # seconds -> microseconds

    # Call the SDK function (4×N pointcloud, reflectance)
    pointcloud, reflectance = build_pointcloud(
        lidar_dir, poses_path, args.extrinsics_dir, start_time, end_time
    )

    # Keep only XYZ (first 3 rows); drop homogeneous row that SDK includes
    points = pointcloud[:3, :].T  # (3,N) -> (N,3)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    np.savez(args.output, points=points, reflectance=reflectance)
    print(f"Built pointcloud with {points.shape[0]} points.")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()

