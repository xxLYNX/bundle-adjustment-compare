#!/usr/bin/env python3
import os
import sys
import argparse
import numpy as np

# --- Ensure SDK python modules are on sys.path (same style as your scripts) ---
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SDK_PY = os.path.join(os.path.dirname(THIS_DIR), "robotcar-dataset-sdk-3.1", "python")
if SDK_PY not in sys.path:
    sys.path.insert(0, SDK_PY)

from transform import build_se3_transform
from interpolate_poses import interpolate_ins_poses
from camera_model import CameraModel
from image import load_image  # SDK image loader

import cv2  # pip install opencv-python

def load_timestamps(ts_path):
    """Read timestamps from a timestamp file
    Returns: np.array of int64 timestamps
    """
    if not os.path.exists(ts_path):
        raise FileNotFoundError(f"Timestamp file not found: {ts_path}")
    
    stamps = []
    with open(ts_path) as f:
        for line in f:
            stamps.append(int(line.split()[0]))
    return np.array(stamps, dtype=np.int64)

def build_ins_camera_extrinsic(extrinsics_dir, camera_stream):
    # Vehicle -> INS
    with open(os.path.join(extrinsics_dir, "ins.txt")) as f:
        veh_ins = [float(x) for x in next(f).split()]
    T_veh_ins = build_se3_transform(veh_ins)

    # Vehicle -> camera
    # Handle stereo cameras: stereo/left and stereo/right both use stereo.txt
    if camera_stream.startswith('stereo/'):
        cam_extr_file = "stereo.txt"
    else:
        cam_extr_file = camera_stream.replace('/', '_') + ".txt"
    with open(os.path.join(extrinsics_dir, cam_extr_file)) as f:
        veh_cam = [float(x) for x in next(f).split()]
    T_veh_cam = build_se3_transform(veh_cam)

    # INS -> camera
    return np.linalg.solve(T_veh_ins, T_veh_cam)

def nearest_indices(src_ts, dst_ts):
    """
    For each src timestamp, find index of nearest dst timestamp.
    Assumes both are sorted ascending.
    """
    idx = np.searchsorted(dst_ts, src_ts)
    idx = np.clip(idx, 1, len(dst_ts) - 1)
    left = idx - 1
    right = idx
    choose_right = (np.abs(dst_ts[right] - src_ts) < np.abs(dst_ts[left] - src_ts))
    return np.where(choose_right, right, left)

def se3_inv(T):
    R = T[:3, :3]
    t = T[:3, 3:4]  # Keep as column vector for matrix multiplication
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3:4] = -R.T @ t  # Assign as column vector
    return Ti

def rot_to_quat_wxyz(R):
    # Returns (qw, qx, qy, qz)
    # Minimal, stable enough for logging/initialization.
    qw = np.sqrt(max(0.0, 1.0 + R[0,0] + R[1,1] + R[2,2])) / 2.0
    qx = np.sqrt(max(0.0, 1.0 + R[0,0] - R[1,1] - R[2,2])) / 2.0
    qy = np.sqrt(max(0.0, 1.0 - R[0,0] + R[1,1] - R[2,2])) / 2.0
    qz = np.sqrt(max(0.0, 1.0 - R[0,0] - R[1,1] + R[2,2])) / 2.0
    qx = np.copysign(qx, R[2,1] - R[1,2])
    qy = np.copysign(qy, R[0,2] - R[2,0])
    qz = np.copysign(qz, R[1,0] - R[0,1])
    return qw, qx, qy, qz

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", required=True)
    ap.add_argument("--extrinsics_dir", required=True)
    ap.add_argument("--models_dir", required=True)
    ap.add_argument("--poses_file", default="gps/ins.csv")
    ap.add_argument("--left_stream", default="mono_left")
    ap.add_argument("--right_stream", default="mono_right")
    ap.add_argument("--left_timestamps", required=True, help="Path to left camera timestamps file")
    ap.add_argument("--right_timestamps", required=True, help="Path to right camera timestamps file")
    ap.add_argument("--duration", type=float, default=10.0)
    ap.add_argument("--frame_stride", type=int, default=2)
    ap.add_argument("--max_frames", type=int, default=10)
    ap.add_argument("--min_disparity_px", type=float, default=2.0)
    ap.add_argument("--max_points", type=int, default=50000)
    ap.add_argument("--output_prefix", required=True)
    args = ap.parse_args()

    dataset_root = args.dataset_root
    poses_path = os.path.join(dataset_root, args.poses_file)

    # Camera models (use SDK intrinsics)
    left_images_dir = os.path.join(dataset_root, args.left_stream)
    right_images_dir = os.path.join(dataset_root, args.right_stream)
    camL = CameraModel(args.models_dir, left_images_dir)
    camR = CameraModel(args.models_dir, right_images_dir)

    fx, fy = camL.focal_length
    cx, cy = camL.principal_point
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0,  0,  1]], dtype=np.float64)

    # Extrinsics: INS -> Stereo Rig (from stereo.txt)
    T_ins_rig = build_ins_camera_extrinsic(args.extrinsics_dir, args.left_stream)
    
    # Apply G_camera_image to get individual camera transforms
    # G_camera_image transforms from image frame to camera frame
    T_ins_L = T_ins_rig @ np.linalg.inv(np.asarray(camL.G_camera_image))
    T_ins_R = T_ins_rig @ np.linalg.inv(np.asarray(camR.G_camera_image))

    # Time window based on left timestamps
    tsL_all = load_timestamps(args.left_timestamps)
    tsR_all = load_timestamps(args.right_timestamps)

    origin_time = int(tsL_all[0])
    end_time = origin_time + int(args.duration * 1e6)
    mask = (tsL_all >= origin_time) & (tsL_all <= end_time)
    tsL = tsL_all[mask][::args.frame_stride][:args.max_frames]

    if len(tsL) == 0:
        raise RuntimeError("No left frames in requested window.")

    # Pair each left frame with nearest right frame
    idxR = nearest_indices(tsL, tsR_all)
    tsR = tsR_all[idxR]

    print(f"Using {len(tsL)} left frames from {args.left_stream} and nearest-matched right frames from {args.right_stream}.")

    # INS poses at those timestamps (gives pose matrices compatible with SDK conventions)
    poses_ins_list = interpolate_ins_poses(poses_path, tsL.tolist(), origin_time, use_rtk=False)
    poses_ins = [np.asarray(T) for T in poses_ins_list]  # avoid numpy.matrix surprises
    # ORB + matcher
    orb = cv2.ORB_create(nfeatures=3000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    # Global BA data
    points_w = []                  # list of (x,y,z)
    observations = []              # list of (frame_id, pid, u, v)
    pid_count = 0

    # Track state from previous left frame
    prev_desc = None
    prev_kps = None
    prev_kp_to_pid = {}            # keypoint index -> pid

    # World-to-camera pose lines (match your existing ba_module expectations)
    pose_lines = []

    for i, (tL, tR) in enumerate(zip(tsL, tsR)):
        # Load images via SDK loader (undistorted); then convert to grayscale uint8
        imgL = load_image(os.path.join(left_images_dir, f"{tL}.png"), camL)
        imgR = load_image(os.path.join(right_images_dir, f"{tR}.png"), camR)

        if imgL.ndim == 3:
            grayL = cv2.cvtColor(imgL, cv2.COLOR_RGB2GRAY)
        else:
            grayL = imgL.astype(np.uint8)

        if imgR.ndim == 3:
            grayR = cv2.cvtColor(imgR, cv2.COLOR_RGB2GRAY)
        else:
            grayR = imgR.astype(np.uint8)

        kpsL, desL = orb.detectAndCompute(grayL, None)
        kpsR, desR = orb.detectAndCompute(grayR, None)

        if desL is None or desR is None or len(kpsL) < 20 or len(kpsR) < 20:
            print(f"Frame {i}: insufficient features, skipping.")
            prev_desc, prev_kps, prev_kp_to_pid = desL, kpsL, {}
            continue

        # --- Step A: temporal matching (left_t-1 -> left_t) to keep track ids alive
        kp_to_pid = {}
        if prev_desc is not None and prev_kps is not None and len(prev_kps) > 0:
            m01 = bf.knnMatch(prev_desc, desL, k=2)
            for m, n in m01:
                if m.distance < 0.75 * n.distance:
                    prev_idx = m.queryIdx
                    cur_idx = m.trainIdx
                    if prev_idx in prev_kp_to_pid:
                        kp_to_pid[cur_idx] = prev_kp_to_pid[prev_idx]

        # --- Step B: stereo matching (left_t -> right_t) for triangulation / new points
        mLR = bf.knnMatch(desL, desR, k=2)
        good = []
        for m, n in mLR:
            if m.distance < 0.75 * n.distance:
                uL, vL = kpsL[m.queryIdx].pt
                uR, vR = kpsR[m.trainIdx].pt
                # quick disparity sanity (works best if roughly rectified)
                if (uL - uR) >= args.min_disparity_px and abs(vL - vR) < 3.0:
                    good.append(m)

        # Build camera pose for this frame from INS (INS->cam extrinsic + INS pose)
        T_ins_w = poses_ins[i]              # SDK returns INS-to-world (camera-to-world)
        T_L_w = T_ins_w @ T_ins_L           # Left camera-to-world
        T_R_w = T_ins_w @ T_ins_R           # Right camera-to-world

        # Left->Right relative pose (in left camera coords)
        T_L_R = se3_inv(T_L_w) @ T_R_w
        R = T_L_R[:3, :3]
        t = T_L_R[:3, 3:4]

        P1 = K @ np.hstack([np.eye(3), np.zeros((3,1))])
        P2 = K @ np.hstack([R, t])

        # Prepare triangulation points
        ptsL = []
        ptsR = []
        lr_pairs = []
        for m in good:
            cur_idx = m.queryIdx
            uL, vL = kpsL[cur_idx].pt
            uR, vR = kpsR[m.trainIdx].pt
            ptsL.append([uL, vL])
            ptsR.append([uR, vR])
            lr_pairs.append((cur_idx, uL, vL))

        if len(ptsL) > 0:
            ptsL = np.array(ptsL, dtype=np.float64).T  # 2xN
            ptsR = np.array(ptsR, dtype=np.float64).T  # 2xN
            X_h = cv2.triangulatePoints(P1, P2, ptsL, ptsR)  # 4xN
            X = (X_h[:3, :] / X_h[3:4, :]).T  # Nx3 in LEFT cam coords

            # Convert LEFT-cam points to world
            for j, (cur_idx, uL, vL) in enumerate(lr_pairs):
                if len(points_w) >= args.max_points:
                    break

                Xc = X[j]
                if not np.isfinite(Xc).all():
                    continue
                if Xc[2] <= 0.1:
                    continue

                # If this keypoint already has a track id, don't spawn a new 3D point.
                if cur_idx in kp_to_pid:
                    pid = kp_to_pid[cur_idx]
                else:
                    # New point id
                    pid = pid_count
                    pid_count += 1
                    Xw_h = np.array([Xc[0], Xc[1], Xc[2], 1.0], dtype=np.float64)
                    Xw = np.asarray(T_L_w @ Xw_h).ravel()   # Transform from camera to world
                    points_w.append((float(Xw[0]), float(Xw[1]), float(Xw[2])))
                    kp_to_pid[cur_idx] = pid

                observations.append((i, pid, float(uL), float(vL)))

        print(f"Frame {i}: {len(observations)} total observations so far, {len(points_w)} points so far.")

        # --- Pose output: world-to-camera transform for Ceres ---
        # Ceres expects: p_camera = R * p_world + t
        T_w_cam = se3_inv(T_L_w)  # Invert camera-to-world to get world-to-camera
        qw, qx, qy, qz = rot_to_quat_wxyz(T_w_cam[:3, :3])
        t_vec = np.asarray(T_w_cam[:3, 3]).ravel()  # Kill numpy.matrix, ensure 1D
        tx, ty, tz = float(t_vec[0]), float(t_vec[1]), float(t_vec[2])
        # Format: tx ty tz qx qy qz qw (translation first, then quaternion)
        pose_lines.append((tx, ty, tz, qx, qy, qz, qw))

        prev_desc, prev_kps, prev_kp_to_pid = desL, kpsL, kp_to_pid

    out_poses = args.output_prefix + "_poses.txt"
    out_points = args.output_prefix + "_points.txt"
    out_obs = args.output_prefix + "_observations.txt"

    with open(out_poses, "w") as f:
        for i, (tx, ty, tz, qx, qy, qz, qw) in enumerate(pose_lines):
            f.write(f"{i} {tx:.12g} {ty:.12g} {tz:.12g} {qx:.12g} {qy:.12g} {qz:.12g} {qw:.12g}\n")

    with open(out_points, "w") as f:
        for pid, (x, y, z) in enumerate(points_w):
            f.write(f"{pid} {x:.12g} {y:.12g} {z:.12g}\n")

    with open(out_obs, "w") as f:
        for (fid, pid, u, v) in observations:
            f.write(f"{fid} {pid} {u:.6f} {v:.6f}\n")

    print("Wrote:")
    print(" ", out_poses)
    print(" ", out_points)
    print(" ", out_obs)

if __name__ == "__main__":
    main()

