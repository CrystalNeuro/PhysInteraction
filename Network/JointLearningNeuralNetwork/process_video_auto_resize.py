#!/usr/bin/env python3
"""
ULTIMATE version: Auto-resizes images on-the-fly!
No need for manual conversion - just point it at your dataset folder.

Usage:
    python process_video_auto_resize.py \
        --depth_frames_dir /path/to/dataset/841412060263 \
        --output_dir outputs \
        --max_frames 100
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


def auto_resize_depth(depth_img, target_size=(240, 320)):
    """Automatically resize depth image to target size if needed."""
    h, w = depth_img.shape[:2]
    if (h, w) == target_size:
        return depth_img
    
    print(f"    Auto-resizing from {h}×{w} to {target_size[0]}×{target_size[1]}")
    return cv2.resize(depth_img, (target_size[1], target_size[0]), interpolation=cv2.INTER_NEAREST)


def validate_joints(joints_xyz, frame_idx=None):
    """Validate if joints are in reasonable range."""
    prefix = f"[Frame {frame_idx}] " if frame_idx is not None else ""
    
    if np.any(joints_xyz[:, 2] < 50):
        return False, f"{prefix}Joints too close (z < 50mm)"
    if np.any(joints_xyz[:, 2] > 5000):
        return False, f"{prefix}Joints too far (z > 5000mm)"
    if not np.all(np.isfinite(joints_xyz)):
        return False, f"{prefix}Joints contain NaN/Inf"
    
    bbox_size = joints_xyz.max(axis=0) - joints_xyz.min(axis=0)
    if np.any(bbox_size > 600):
        return False, f"{prefix}Hand bbox too large: {bbox_size}"
    if np.any(bbox_size < 10):
        return False, f"{prefix}Hand bbox too small: {bbox_size}"
    
    return True, "OK"


def post_depth_to_server(depth_img_uint16, server_url="http://127.0.0.1:8080", begin_flag=1, depth_scale=1.0):
    """Send depth frame to inference server."""
    if depth_scale != 1.0:
        depth_img_uint16 = (depth_img_uint16.astype(np.float32) * depth_scale)
        depth_img_uint16 = np.clip(depth_img_uint16, 0, 65535).astype(np.uint16)

    # Auto-resize to 240×320
    depth_img_uint16 = auto_resize_depth(depth_img_uint16, target_size=(240, 320))

    # Encode PNG
    ok, buf = cv2.imencode('.png', depth_img_uint16)
    if not ok:
        raise RuntimeError("PNG encoding failed")
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
    """Find matching color frame."""
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
    globs = glob.glob(os.path.join(color_dir, f"*{idx}*.png")) + glob.glob(os.path.join(color_dir, f"*{idx}*.jpg"))
    return globs[0] if globs else None


def create_overlay_visualization(frame_idx, output_dir, depth_img, joints_uvz, mask):
    """Create overlay visualization."""
    base = os.path.join(output_dir, f"frame_{frame_idx:06d}")
    overlay_path = base + "_overlay.png"
    
    # Resize depth for visualization to match joint coordinates (240×320)
    depth_resized = auto_resize_depth(depth_img, target_size=(240, 320))
    
    if depth_resized.dtype == np.uint16:
        depth_vis = cv2.normalize(depth_resized, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    else:
        depth_vis = depth_resized
    
    depth_rgb = cv2.cvtColor(depth_vis, cv2.COLOR_GRAY2BGR)
    
    # Overlay mask
    if mask is not None:
        hand_mask = (mask > 128).astype(np.uint8)
        depth_rgb[:, :, 1] = np.maximum(depth_rgb[:, :, 1], hand_mask * 100)
    
    # Draw joints
    joint_colors = [
        (0, 0, 255),    # Wrist - red
        (255, 0, 0), (255, 0, 0), (255, 0, 0), (255, 0, 0),  # Thumb - blue
        (0, 255, 0), (0, 255, 0), (0, 255, 0), (0, 255, 0),  # Index - green
        (255, 255, 0), (255, 255, 0), (255, 255, 0), (255, 255, 0),  # Middle - cyan
        (255, 0, 255), (255, 0, 255), (255, 0, 255), (255, 0, 255),  # Ring - magenta
        (0, 255, 255), (0, 255, 255), (0, 255, 255), (0, 255, 255),  # Pinky - yellow
    ]
    
    for i, (u, v, z) in enumerate(joints_uvz):
        u_int, v_int = int(v), int(u)
        if 0 <= u_int < depth_rgb.shape[0] and 0 <= v_int < depth_rgb.shape[1]:
            color = joint_colors[i]
            cv2.circle(depth_rgb, (v_int, u_int), 4, color, -1)
            if i == 0:
                cv2.circle(depth_rgb, (v_int, u_int), 8, (0, 0, 255), 2)
    
    # Draw skeleton
    connections = [
        [0, 1], [1, 2], [2, 3], [3, 4],  # Thumb
        [0, 5], [5, 6], [6, 7], [7, 8],  # Index
        [0, 9], [9, 10], [10, 11], [11, 12],  # Middle
        [0, 13], [13, 14], [14, 15], [15, 16],  # Ring
        [0, 17], [17, 18], [18, 19], [19, 20]   # Pinky
    ]
    for i, j in connections:
        u1, v1, _ = joints_uvz[i]
        u2, v2, _ = joints_uvz[j]
        u1_int, v1_int = int(v1), int(u1)
        u2_int, v2_int = int(v2), int(u2)
        if (0 <= u1_int < depth_rgb.shape[0] and 0 <= v1_int < depth_rgb.shape[1] and
            0 <= u2_int < depth_rgb.shape[0] and 0 <= v2_int < depth_rgb.shape[1]):
            cv2.line(depth_rgb, (v1_int, u1_int), (v2_int, u2_int), (255, 255, 255), 2)
    
    cv2.imwrite(overlay_path, depth_rgb)
    return overlay_path


def save_outputs(frame_idx, output_dir, mano, joints_xyz, joints_uvz, mask, pose_fit, 
                depth_fp=None, depth_img=None, color_fp=None, render_3d=True):
    """Save all outputs for a frame."""
    pose_pca, global_R, global_T, error = pose_fit
    vertices = mano.get_posed_model(pose_pca, global_R, global_T)

    base = os.path.join(output_dir, f"frame_{frame_idx:06d}")
    mesh_path = base + "_hand_mesh.obj"
    joints_path = base + "_hand_mesh_joints.obj"
    mask_path = base + "_mask.png"
    depth_out = base + "_depth.png"
    color_out = base + "_color.png"

    mano.export_obj(mesh_path, vertices)

    with open(joints_path, 'w') as f:
        f.write("# 21 Hand Joints\n")
        for xyz in joints_xyz:
            f.write(f"v {xyz[0]:.4f} {xyz[1]:.4f} {xyz[2]:.4f}\n")

    cv2.imwrite(mask_path, mask)

    try:
        if depth_fp and os.path.exists(depth_fp):
            shutil.copyfile(depth_fp, depth_out)
        elif depth_img is not None:
            cv2.imwrite(depth_out, depth_img)
    except Exception as e:
        pass

    try:
        if color_fp and os.path.exists(color_fp):
            shutil.copyfile(color_fp, color_out)
    except Exception as e:
        pass

    overlay_path = create_overlay_visualization(frame_idx, output_dir, depth_img, joints_uvz, mask)

    if render_3d:
        try:
            render_out = base + "_render.png"
            vtx, faces = viz_load_obj(mesh_path)
            joints_arr = None
            try:
                joints_arr, _ = viz_load_obj(joints_path)
            except Exception:
                pass
            viz_render(vtx, faces, joints=joints_arr, output_path=render_out, view_angle=(30, 45))
        except Exception as e:
            pass

    return mesh_path, joints_path, mask_path, overlay_path, error


def process_depth_sequence(depth_iter, output_dir, mano_path, server_url, max_frames=None, 
                          depth_scale=1.0, color_dir=None, 
                          fx=475.0, fy=475.0, cx=160.0, cy=120.0,
                          render_3d=False, max_fit_iter=500):
    """Process a sequence of depth frames."""
    ensure_dir(output_dir)

    mano = MANOModel(mano_path)
    shape_params = np.array([
        -2.61435056, -1.16743336, -2.80988378, 0.12670897, -0.08323125,
        2.28185672, -0.05833138, -2.95105206, -3.43976417, 0.30667237
    ])
    mano.init_rest_model(shape_params)

    count = 0
    errors = []
    successes = 0
    server_errors = 0
    validation_failures = 0
    first_flag = 1
    
    for item in depth_iter:
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
            print(f"[Frame {frame_idx:06d}] ✗ Server error: {e}")
            continue
        finally:
            first_flag = 0

        # Convert to XYZ (in meters for MANO)
        joints_xyz_mm = convert_uvz_to_xyz(joints_uvz, depth_img, fx=fx, fy=fy, cx=cx, cy=cy)
        joints_xyz_m = joints_xyz_mm / 1000.0  # Convert to meters

        # Validate
        is_valid, msg = validate_joints(joints_xyz_mm, frame_idx)
        if not is_valid:
            validation_failures += 1
            print(f"[Frame {frame_idx:06d}] ⚠ Validation failed: {msg}")
            create_overlay_visualization(frame_idx, output_dir, depth_img, joints_uvz, mask)
            continue

        # Fit MANO
        pose_fit = mano.fit_to_joints(joints_xyz_m, max_iter=max_fit_iter)

        # Find color
        color_fp = None
        if color_dir is None and depth_fp is not None:
            color_dir_eff = os.path.dirname(depth_fp)
        else:
            color_dir_eff = color_dir
        if depth_fp is not None and color_dir_eff:
            color_fp = find_color_for_depth(depth_fp, color_dir_eff)

        mesh_path, joints_path, mask_path, overlay_path, err = save_outputs(
            frame_idx, output_dir, mano, joints_xyz_m, joints_uvz, mask, pose_fit,
            depth_fp=depth_fp, depth_img=depth_img, color_fp=color_fp, render_3d=render_3d
        )
        
        # Convert error back to mm² for reporting
        err_mm2 = err * 1000000
        errors.append(float(err_mm2))
        successes += 1

        if err_mm2 < 100000:
            status = "✓ GOOD"
        elif err_mm2 < 300000:
            status = "~ OK"
        else:
            status = "✗ HIGH"

        print(f"[Frame {frame_idx:06d}] {status} | Error: {err_mm2:10.2f} mm² | {os.path.basename(overlay_path)}")

    # Summary
    summary = {
        "frames_processed": count,
        "successes": successes,
        "validation_failures": validation_failures,
        "server_errors": server_errors,
        "mean_fit_error": float(np.mean(errors)) if errors else None,
        "median_fit_error": float(np.median(errors)) if errors else None,
        "min_fit_error": float(np.min(errors)) if errors else None,
        "max_fit_error": float(np.max(errors)) if errors else None,
        "camera_intrinsics": {"fx": fx, "fy": fy, "cx": cx, "cy": cy},
        "output_dir": os.path.abspath(output_dir)
    }
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"Frames processed:     {count}")
    print(f"Successes:            {successes}")
    print(f"Validation failures:  {validation_failures}")
    print(f"Server errors:        {server_errors}")
    if errors:
        print(f"Mean fit error:       {np.mean(errors):.2f} mm²")
        print(f"Median fit error:     {np.median(errors):.2f} mm²")
        print(f"Min/Max fit error:    {np.min(errors):.2f} / {np.max(errors):.2f} mm²")
    print(f"Output directory:     {os.path.abspath(output_dir)}")
    print(f"{'='*80}")


def iter_depth_pngs(depth_dir):
    """Iterate over depth PNGs in directory."""
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
    
    print(f"Found {len(files)} depth images in {depth_dir}")
    
    for i, fp in enumerate(files):
        img = cv2.imread(fp, cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        yield i, img, fp


def main():
    parser = argparse.ArgumentParser(description="Process depth frames with AUTO-RESIZE")
    parser.add_argument('--depth_frames_dir', type=str, required=True, help='Directory of depth PNGs (any size!)')
    parser.add_argument('--output_dir', type=str, default='outputs_auto', help='Output directory')
    parser.add_argument('--mano_path', type=str, default='../../InteractionReconstruction/InteractionReconstruction/data/mano/mano_r.json')
    parser.add_argument('--server_url', type=str, default='http://127.0.0.1:8080')
    parser.add_argument('--max_frames', type=int, default=None, help='Limit frames')
    parser.add_argument('--depth_scale', type=float, default=1.0)
    parser.add_argument('--color_frames_dir', type=str, default=None)
    parser.add_argument('--fx', type=float, default=475.0)
    parser.add_argument('--fy', type=float, default=475.0)
    parser.add_argument('--cx', type=float, default=160.0)
    parser.add_argument('--cy', type=float, default=120.0)
    parser.add_argument('--max_fit_iter', type=int, default=500)
    parser.add_argument('--render_3d', action='store_true', help='Enable slow 3D rendering')

    args = parser.parse_args()

    if not os.path.exists(args.mano_path):
        raise FileNotFoundError(f"MANO model not found: {args.mano_path}")

    print(f"\n{'='*80}")
    print(f"AUTO-RESIZE PROCESSING")
    print(f"{'='*80}")
    print(f"Input directory:  {args.depth_frames_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Camera: fx={args.fx}, fy={args.fy}, cx={args.cx}, cy={args.cy}")
    print(f"Images will be auto-resized to 240×320 if needed")
    print(f"{'='*80}\n")

    if not os.path.isdir(args.depth_frames_dir):
        raise FileNotFoundError(f"Directory not found: {args.depth_frames_dir}")
    
    depth_iter = iter_depth_pngs(args.depth_frames_dir)
    color_dir = args.color_frames_dir or args.depth_frames_dir

    process_depth_sequence(depth_iter, args.output_dir, args.mano_path, args.server_url, 
                          args.max_frames, depth_scale=args.depth_scale, color_dir=color_dir,
                          fx=args.fx, fy=args.fy, cx=args.cx, cy=args.cy,
                          render_3d=args.render_3d, max_fit_iter=args.max_fit_iter)


if __name__ == '__main__':
    main()
