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

from transform import build_se3_transform
from interpolate_poses import interpolate_ins_poses
from camera_model import CameraModel
from image import load_image


def load_lidar_origin_time(dataset_root, lidar_name):
    """
    Recompute the origin_time we used for the pointcloud:
    first timestamp in <lidar>.timestamps.
    """
    ts_path = os.path.join(dataset_root, lidar_name + ".timestamps")
    with open(ts_path) as f:
        first_line = next(f)
    return int(first_line.split()[0])


def load_camera_timestamps(dataset_root, camera_stream):
    """
    camera_stream examples:
      'mono_left'  -> sample/mono_left.timestamps
      'stereo'     -> sample/stereo.timestamps
    """
    ts_path = os.path.join(dataset_root, camera_stream + ".timestamps")
    stamps = []
    with open(ts_path) as f:
        for line in f:
            t = int(line.split()[0])
            stamps.append(t)
    return np.array(stamps, dtype=np.int64)


def build_ins_camera_extrinsic(extrinsics_dir, camera_stream):
    """
    Build transform T_ins_cam (4x4) from INS frame to camera frame,
    using extrinsics text files in extrinsics_dir.

    Each file is a single line: x y z roll pitch yaw (vehicle->sensor).
    """
    # Vehicle -> INS
    with open(os.path.join(extrinsics_dir, "ins.txt")) as f:
        veh_ins = [float(x) for x in next(f).split()]
    T_veh_ins = build_se3_transform(veh_ins)

    # Vehicle -> camera
    cam_extr_file = camera_stream + ".txt"  # e.g., 'mono_left.txt'
    with open(os.path.join(extrinsics_dir, cam_extr_file)) as f:
        veh_cam = [float(x) for x in next(f).split()]
    T_veh_cam = build_se3_transform(veh_cam)

    # INS -> camera:  T_ins_cam = inv(T_veh_ins) * T_veh_cam
    T_ins_cam = np.linalg.solve(T_veh_ins, T_veh_cam)
    return T_ins_cam


def main():
    parser = argparse.ArgumentParser(
        description="Create a BA dataset (poses, points, observations) from RobotCar sample."
    )
    parser.add_argument("--dataset_root", required=True,
                        help="Path to traversal root (where lms_front, mono_left, etc. live).")
    parser.add_argument("--extrinsics_dir", required=True,
                        help="Path to SDK 'extrinsics' directory.")
    parser.add_argument("--models_dir", required=True,
                        help="Path to SDK 'models' directory.")
    parser.add_argument("--pointcloud_npz", required=True,
                        help="Path to pointcloud.npz produced earlier.")
    parser.add_argument("--lidar", default="lms_front",
                        help="Lidar stream name used for the pointcloud (default: lms_front).")
    parser.add_argument("--camera_stream", default="mono_left",
                        help="Camera stream name (e.g. mono_left, mono_right, stereo).")
    parser.add_argument("--poses_file", default="gps/ins.csv",
                        help="Poses file relative to dataset_root (e.g. gps/ins.csv).")
    parser.add_argument("--duration", type=float, default=None,
                        help="If set, clamp camera timestamps to "
                             "[origin_time, origin_time + duration] seconds.")
    parser.add_argument("--frame_stride", type=int, default=20,
                        help="Use every Nth camera frame.")
    parser.add_argument("--max_frames", type=int, default=50,
                        help="Maximum number of frames to include.")
    parser.add_argument("--min_depth", type=float, default=1.0,
                        help="Minimum point depth in camera frame (m).")
    parser.add_argument("--max_depth", type=float, default=80.0,
                        help="Maximum point depth in camera frame (m).")
    parser.add_argument("--output_prefix", default="ba_data",
                        help="Prefix for output files (poses.txt, points.txt, observations.txt).")
    args = parser.parse_args()

    dataset_root = args.dataset_root
    poses_path = os.path.join(dataset_root, args.poses_file)

    # --- Camera model & intrinsics (SDK class, unmodified) ---
    images_dir = os.path.join(dataset_root, args.camera_stream)
    cam_model = CameraModel(args.models_dir, images_dir)
    fx, fy = cam_model.focal_length
    cx, cy = cam_model.principal_point  # from camera_model.py

    # --- Load global point cloud ---
    pc = np.load(args.pointcloud_npz)
    points_world = pc["points"]  # (N, 3)
    num_points = points_world.shape[0]
    print(f"Loaded {num_points} world points from {args.pointcloud_npz}")

    # --- Get origin time consistent with pointcloud ---
    origin_time = load_lidar_origin_time(dataset_root, args.lidar)
    print(f"Origin time (µs) from {args.lidar}.timestamps: {origin_time}")

    # --- Camera timestamps & time-window clamping ---
    cam_ts_all = load_camera_timestamps(dataset_root, args.camera_stream)

    # Lidar time span
    lidar_ts_path = os.path.join(dataset_root, args.lidar + ".timestamps")
    with open(lidar_ts_path) as f:
        last_line = None
        for line in f:
            last_line = line
    lidar_last = int(last_line.split()[0])

    if args.duration is not None:
        requested_end = origin_time + int(args.duration * 1e6)
        end_time = min(requested_end, lidar_last)
        effective_dur_s = (end_time - origin_time) / 1e6
        if requested_end > lidar_last:
            print(f"[WARN] Requested duration {args.duration:.2f}s exceeds lidar data span; "
                  f"clamping to {effective_dur_s:.2f}s.")
        else:
            print(f"Using duration {effective_dur_s:.2f}s for BA time window.")
    else:
        end_time = lidar_last
        effective_dur_s = (end_time - origin_time) / 1e6
        print(f"No duration specified; using full lidar span ({effective_dur_s:.2f}s).")

    valid_mask = (cam_ts_all >= origin_time) & (cam_ts_all <= end_time)
    cam_ts_valid = cam_ts_all[valid_mask]
    cam_ts_sub = cam_ts_valid[::args.frame_stride][:args.max_frames]
    num_frames = len(cam_ts_sub)
    print(f"Using {num_frames} camera frames from {args.camera_stream}")

    if num_frames == 0:
        raise RuntimeError(
            "No camera timestamps fall inside selected time window; "
            "try reducing duration or adjusting filters."
        )
    if num_frames < args.max_frames:
        print(f"[WARN] Requested max_frames={args.max_frames}, but only {num_frames} "
              f"frames are available in the selected time window.")

    # --- Get image size from the first frame ---
    first_ts = int(cam_ts_sub[0])

    # Images are stored as <timestamp>.png under the camera directory
    image_path = os.path.join(images_dir, f"{first_ts}.png")
    if not os.path.exists(image_path):
        # Fallback if any sequence uses JPG instead of PNG
        jpg_path = os.path.join(images_dir, f"{first_ts}.jpg")
        if os.path.exists(jpg_path):
            image_path = jpg_path
        else:
            raise FileNotFoundError(f"Could not find image for timestamp {first_ts} "
                                    f"at {image_path} or {jpg_path}")

    # Correct usage: path first, then optional model
    img = load_image(image_path, cam_model)
    height, width = img.shape[0], img.shape[1]
    image_size = (height, width)
    print(f"Image size: {width}x{height}")

    # --- INS poses at those timestamps (world <- INS) ---
    poses_ins_list = interpolate_ins_poses(
        poses_path,
        cam_ts_sub.tolist(),
        origin_time,
        use_rtk=False
    )
    # SDK returns numpy.matrix; convert to ndarray so we can stack into (F,4,4)
    poses_ins = np.stack([np.asarray(T) for T in poses_ins_list], axis=0)  # (F, 4, 4)

    # --- INS -> camera extrinsic ---
    T_ins_cam = build_ins_camera_extrinsic(args.extrinsics_dir, args.camera_stream)

    # --- Precompute homogeneous world points ---
    N = num_points
    Xw_h = np.vstack([points_world.T, np.ones((1, N))])  # (4, N)

    # --- Containers for BA outputs ---
    poses_out = []       # rows: frame_id tx ty tz qx qy qz qw
    points_out = points_world  # rows: point_id X Y Z
    obs = []             # rows: frame_id point_id u v

    for frame_id, (t, T_w_ins) in enumerate(zip(cam_ts_sub, poses_ins)):
        # World -> camera transform
        T_w_cam = T_w_ins @ T_ins_cam
        T_cam_w = np.linalg.inv(T_w_cam)

        # Transform all points into camera frame
        Xc_h = T_cam_w @ Xw_h
        Xc_h = np.asarray(Xc_h)  # ensure ndarray, not numpy.matrix
        Xc = Xc_h[:3, :]         # (3, N)
        x = Xc[0, :]
        y = Xc[1, :]
        z = Xc[2, :]

        # Depth & FOV filtering
        valid = (z > args.min_depth) & (z < args.max_depth)

        # Project with pinhole intrinsics from CameraModel
        u = fx * (x / z) + cx
        v = fy * (y / z) + cy

        # Inside-image check; CameraModel.project uses 0.5 .. width/height
        valid &= (u >= 0.5) & (u <= image_size[1]) \
               & (v >= 0.5) & (v <= image_size[0])

        idxs = np.nonzero(valid)[0]
        print(f"Frame {frame_id}: {len(idxs)} visible points")

        for pid in idxs:
            obs.append((frame_id, pid, float(u[pid]), float(v[pid])))

        # Record pose (world->cam) as translation + quaternion
        R = T_cam_w[:3, :3]
        tvec = np.asarray(T_cam_w[:3, 3]).ravel()  # flatten to avoid DeprecationWarning
        qw = np.sqrt(max(0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2.0
        qx = (R[2, 1] - R[1, 2]) / (4 * qw)
        qy = (R[0, 2] - R[2, 0]) / (4 * qw)
        qz = (R[1, 0] - R[0, 1]) / (4 * qw)

        poses_out.append((frame_id,
                          float(tvec[0]), float(tvec[1]), float(tvec[2]),
                          float(qx), float(qy), float(qz), float(qw)))

    # --- Save everything ---
    os.makedirs(os.path.dirname(args.output_prefix), exist_ok=True)
    prefix = args.output_prefix
    poses_path_out = prefix + "_poses.txt"
    points_path_out = prefix + "_points.txt"
    obs_path_out = prefix + "_observations.txt"

    with open(poses_path_out, "w") as f:
        for row in poses_out:
            f.write(" ".join(str(x) for x in row) + "\n")

    with open(points_path_out, "w") as f:
        for pid, (X, Y, Z) in enumerate(points_out):
            f.write(f"{pid} {X} {Y} {Z}\n")

    with open(obs_path_out, "w") as f:
        for frame_id, pid, u_, v_ in obs:
            f.write(f"{frame_id} {pid} {u_} {v_}\n")

    print(f"Wrote {len(poses_out)} poses to {poses_path_out}")
    print(f"Wrote {points_out.shape[0]} points to {points_path_out}")
    print(f"Wrote {len(obs)} observations to {obs_path_out}")


if __name__ == "__main__":
    main()

