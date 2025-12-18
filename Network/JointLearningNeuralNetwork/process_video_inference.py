#!/usr/bin/env python3
"""
Process a video or a sequence of depth frames for per-frame hand inference.

This script sends each frame to the JointLearningNeuralNetwork inference server,
converts the returned (u, v, z) joints to (x, y, z), fits the MANO model,
and saves per-frame outputs (mesh OBJ, joints OBJ, segmentation mask).

Notes:
- Preferred input is 16-bit aligned depth PNGs (aligned_depth_to_color_*.png).
- Depth MP4 videos are typically 8-bit and lose precision; supported here only
  for demo purposes.

Usage examples:

1) Process original aligned depth PNGs
   python process_video_inference.py \
     --depth_frames_dir /path/to/device/aligned_depth_frames \
     --output_dir /path/to/output \
     --max_frames 150

2) Process a depth video (approximate)
   python process_video_inference.py \
     --depth_video /path/to/depth.mp4 \
     --output_dir /path/to/output \
     --max_frames 150

Make sure the inference server is running:
   cd Network/JointLearningNeuralNetwork
   python inference_server.py --port 8080
"""

import os
import io
import glob
import argparse
import base64
import json
import cv2
import numpy as np
import requests
import shutil
import re

from mano_model import MANOModel
from generate_hand_mesh import convert_uvz_to_xyz
from visualize_mesh import load_obj as viz_load_obj, render_with_matplotlib as viz_render


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def post_depth_to_server(depth_img_uint16, server_url="http://127.0.0.1:8080", begin_flag=1, depth_scale=1.0):
    """
    Encode a single depth frame (uint16 or uint8) to PNG in-memory and POST to server.
    Returns (joints_uvz: np.ndarray[21,3], mask: np.ndarray[H,W]).
    """
    # Apply optional scaling to match server's expected units (server divides by 8 internally)
    if depth_scale != 1.0:
        depth_img_uint16 = (depth_img_uint16.astype(np.float32) * depth_scale)
        depth_img_uint16 = np.clip(depth_img_uint16, 0, 65535).astype(np.uint16)

    # Resize to 240x320 as expected by the server graph
    h, w = depth_img_uint16.shape[:2]
    if (h, w) != (240, 320):
        depth_img_uint16 = cv2.resize(depth_img_uint16, (320, 240), interpolation=cv2.INTER_NEAREST)

    # Encode PNG bytes
    ok, buf = cv2.imencode('.png', depth_img_uint16)
    if not ok:
        raise RuntimeError("PNG encoding failed for depth frame")
    png_bytes = buf.tobytes()

    files = {
        'depth': ('depth.png', io.BytesIO(png_bytes), 'image/png'),
        'begin': ('begin', str(begin_flag).encode('utf-8'), 'text/plain')
    }
    resp = requests.post(server_url, files=files, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"Server error {resp.status_code}: {resp.text[:200]}")

    result = resp.json()
    joints_bytes = base64.b64decode(result['joints'])
    joints_uv_z = np.frombuffer(joints_bytes, dtype=np.float32).reshape(21, 3)

    mask_bytes = base64.b64decode(result['segment'])
    mask_array = np.frombuffer(mask_bytes, dtype=np.uint8)
    mask = cv2.imdecode(mask_array, cv2.IMREAD_UNCHANGED)

    return joints_uv_z, mask


def find_color_for_depth(depth_fp, color_dir):
    """Find matching color frame for a given depth file by index in filename."""
    if not color_dir or not os.path.isdir(color_dir):
        return None
    base = os.path.basename(depth_fp)
    m = re.search(r'(\d{4,})', base)
    if not m:
        return None
    idx = m.group(1)
    candidates = [
        os.path.join(color_dir, f"color_{idx}.png"),
        os.path.join(color_dir, f"rgb_{idx}.png"),
        os.path.join(color_dir, f"color_{idx}.jpg"),
        os.path.join(color_dir, f"rgb_{idx}.jpg"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    # Fallback: glob any file containing the index
    globs = glob.glob(os.path.join(color_dir, f"*{idx}*.png")) + glob.glob(os.path.join(color_dir, f"*{idx}*.jpg"))
    return globs[0] if globs else None


def save_outputs(frame_idx, output_dir, mano, joints_xyz, mask, pose_fit, depth_fp=None, depth_img=None, color_fp=None):
    pose_pca, global_R, global_T, error = pose_fit
    vertices = mano.get_posed_model(pose_pca, global_R, global_T)

    base = os.path.join(output_dir, f"frame_{frame_idx:06d}")
    mesh_path = base + "_hand_mesh.obj"
    joints_path = base + "_hand_mesh_joints.obj"
    mask_path = base + "_mask.png"
    depth_out = base + "_depth.png"
    color_out = base + "_color.png"

    # Export mesh
    mano.export_obj(mesh_path, vertices)

    # Export joints as OBJ points
    with open(joints_path, 'w') as f:
        f.write("# 21 Hand Joints\n")
        for xyz in joints_xyz:
            f.write(f"v {xyz[0]:.4f} {xyz[1]:.4f} {xyz[2]:.4f}\n")

    # Save segmentation mask
    cv2.imwrite(mask_path, mask)

    # Copy or save original depth
    try:
        if depth_fp and os.path.exists(depth_fp):
            shutil.copyfile(depth_fp, depth_out)
        elif depth_img is not None:
            cv2.imwrite(depth_out, depth_img)
    except Exception as e:
        print(f"[Frame {frame_idx}] Warning: failed to save depth image: {e}")

    # Copy color if available
    try:
        if color_fp and os.path.exists(color_fp):
            shutil.copyfile(color_fp, color_out)
    except Exception as e:
        print(f"[Frame {frame_idx}] Warning: failed to save color image: {e}")

    # Render mesh + joints into a PNG image for quick visualization
    try:
        render_out = base + "_render.png"
        vtx, faces = viz_load_obj(mesh_path)
        joints_arr = None
        try:
            joints_arr, _ = viz_load_obj(joints_path)
        except Exception:
            joints_arr = None
        viz_render(vtx, faces, joints=joints_arr, output_path=render_out, view_angle=(30, 45))
    except Exception as e:
        print(f"[Frame {frame_idx}] Warning: failed to render mesh image: {e}")

    return mesh_path, joints_path, mask_path, error


def process_depth_sequence(depth_iter, output_dir, mano_path, server_url, max_frames=None, depth_scale=1.0, color_dir=None):
    ensure_dir(output_dir)

    # Load MANO once
    mano = MANOModel(mano_path)
    # Calibrated shape parameters (from repo)
    shape_params = np.array([
        -2.61435056, -1.16743336, -2.80988378, 0.12670897, -0.08323125,
        2.28185672, -0.05833138, -2.95105206, -3.43976417, 0.30667237
    ])
    mano.init_rest_model(shape_params)

    count = 0
    errors = []
    successes = 0
    server_errors = 0
    first_flag = 1
    for item in depth_iter:
        # depth_iter yields (idx, img, fp) for PNGs, or (idx, frame) for video
        if len(item) == 3:
            frame_idx, depth_img, depth_fp = item
        else:
            frame_idx, depth_img = item
            depth_fp = None
        if max_frames is not None and count >= max_frames:
            break
        count += 1

        try:
            joints_uvz, mask = post_depth_to_server(depth_img, server_url, begin_flag=first_flag, depth_scale=depth_scale)
        except Exception as e:
            server_errors += 1
            print(f"[Frame {frame_idx}] Server error: {e}")
            # after first successful init, use 0; keep 1 until we get a response
            continue
        finally:
            # After first attempted post, set flag to 0 for subsequent frames
            first_flag = 0

        # Convert to XYZ
        joints_xyz = convert_uvz_to_xyz(joints_uvz, depth_img)

        # Fit MANO
        pose_fit = mano.fit_to_joints(joints_xyz, max_iter=200)

        # Find matching color if possible
        color_fp = None
        if color_dir is None and depth_fp is not None:
            # default to same dir as depth
            color_dir_eff = os.path.dirname(depth_fp)
        else:
            color_dir_eff = color_dir
        if depth_fp is not None and color_dir_eff:
            color_fp = find_color_for_depth(depth_fp, color_dir_eff)

        mesh_path, joints_path, mask_path, err = save_outputs(
            frame_idx, output_dir, mano, joints_xyz, mask, pose_fit,
            depth_fp=depth_fp, depth_img=depth_img, color_fp=color_fp
        )
        errors.append(float(err))
        successes += 1

        print(f"[Frame {frame_idx}] Saved: \n  {os.path.basename(mesh_path)}\n  {os.path.basename(joints_path)}\n  {os.path.basename(mask_path)}\n  Fit error: {err:.4f}")

    # Summary JSON
    summary = {
        "frames_processed": count,
        "successes": successes,
        "server_errors": server_errors,
        "mean_fit_error": float(np.mean(errors)) if errors else None,
        "output_dir": os.path.abspath(output_dir)
    }
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nDone. Frames processed: {count}. Successes: {successes}. Server errors: {server_errors}. Summary saved to summary.json")


def iter_depth_pngs(depth_dir):
    # Prefer aligned_depth_to_color_* naming
    patterns = [
        os.path.join(depth_dir, "aligned_depth_to_color_*.png"),
        os.path.join(depth_dir, "depth_*.png"),
        os.path.join(depth_dir, "*.png"),
    ]
    files = []
    for p in patterns:
        files.extend(glob.glob(p))
    files = sorted(set(files))
    if not files:
        raise FileNotFoundError(f"No depth PNGs found in {depth_dir}")
    for i, fp in enumerate(files):
        img = cv2.imread(fp, cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        yield i, img, fp


def iter_depth_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # Convert to grayscale if 3-channel
        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Treat as uint8 depth (demo only)
        frame = frame.astype(np.uint8)
        idx += 1
        yield idx - 1, frame
    cap.release()


def main():
    parser = argparse.ArgumentParser(description="Per-frame hand inference from video or depth frames")
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument('--depth_frames_dir', type=str, help='Directory of aligned depth PNGs')
    g.add_argument('--depth_video', type=str, help='Path to depth video (8-bit, demo only)')
    parser.add_argument('--output_dir', type=str, default='outputs_video_infer', help='Output directory')
    parser.add_argument('--mano_path', type=str, default='../../InteractionReconstruction/InteractionReconstruction/data/mano/mano_r.json', help='Path to MANO model JSON')
    parser.add_argument('--server_url', type=str, default='http://127.0.0.1:8080', help='Inference server URL')
    parser.add_argument('--max_frames', type=int, default=None, help='Limit number of frames processed')
    parser.add_argument('--depth_scale', type=float, default=1.0, help='Multiply input depth before sending to server (use 8.0 if your depth is in mm)')
    parser.add_argument('--color_frames_dir', type=str, default=None, help='Directory of color/RGB frames matching depth indices')

    args = parser.parse_args()

    if not os.path.exists(args.mano_path):
        raise FileNotFoundError(f"MANO model JSON not found: {args.mano_path}")

    if args.depth_frames_dir:
        if not os.path.isdir(args.depth_frames_dir):
            raise FileNotFoundError(f"Depth frames dir not found: {args.depth_frames_dir}")
        depth_iter = iter_depth_pngs(args.depth_frames_dir)
        # Default color dir to depth dir if not provided
        color_dir = args.color_frames_dir or args.depth_frames_dir
    else:
        depth_iter = iter_depth_video(args.depth_video)
        color_dir = args.color_frames_dir

    process_depth_sequence(depth_iter, args.output_dir, args.mano_path, args.server_url, args.max_frames, depth_scale=args.depth_scale, color_dir=color_dir)


if __name__ == '__main__':
    main()