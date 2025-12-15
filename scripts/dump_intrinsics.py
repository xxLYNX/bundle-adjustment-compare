#!/usr/bin/env python3
import os
import sys
import json
import argparse

# Ensure SDK python modules are on sys.path
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SDK_PY = os.path.join(os.path.dirname(THIS_DIR), "robotcar-dataset-sdk-3.1", "python")
if SDK_PY not in sys.path:
    sys.path.insert(0, SDK_PY)

from camera_model import CameraModel  # from SDK


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", required=True,
                        help="Root of traversal (where mono_left, lms_front, etc. live).")
    parser.add_argument("--models_dir", required=True,
                        help="Path to SDK 'models' directory.")
    parser.add_argument("--camera_stream", default="mono_left",
                        help="Camera stream name, e.g. mono_left, stereo, etc.")
    parser.add_argument("--output", required=True,
                        help="Output JSON file for intrinsics.")
    args = parser.parse_args()

    images_dir = os.path.join(args.dataset_root, args.camera_stream)
    cam = CameraModel(args.models_dir, images_dir)

    fx, fy = cam.focal_length
    cx, cy = cam.principal_point

    cfg = {
        "camera_stream": args.camera_stream,
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(cfg, f, indent=2)

    print("Wrote intrinsics to", args.output)


if __name__ == "__main__":
    main()

