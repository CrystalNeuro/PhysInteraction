#!/usr/bin/env python3
"""
Complete pipeline to generate 3D hand mesh from depth image.

1. Send depth image to inference_server.py
2. Get 21 joint positions
3. Fit MANO model to joints
4. Export 3D mesh as OBJ

Usage:
    python generate_hand_mesh.py --depth_image path/to/depth.png --output hand_mesh.obj
"""

import argparse
import requests
import base64
import json
import numpy as np
import cv2
import os

from mano_model import MANOModel


def get_joints_from_server(depth_image_path, server_url="http://127.0.0.1:8080"):
    """Send depth image to inference server and get joint positions."""
    print(f"Sending depth image to server: {server_url}")
    
    with open(depth_image_path, 'rb') as f:
        files = {
            'depth': ('depth.png', f, 'image/png'),
            'begin': ('begin', b'1', 'text/plain')
        }
        response = requests.post(server_url, files=files)
    
    if response.status_code != 200:
        raise Exception(f"Server error: {response.status_code}")
    
    result = response.json()
    
    # Decode joints (21 x 3: u, v, z)
    joints_bytes = base64.b64decode(result['joints'])
    joints_uv_z = np.frombuffer(joints_bytes, dtype=np.float32).reshape(21, 3)
    
    # Decode mask
    mask_bytes = base64.b64decode(result['segment'])
    mask_array = np.frombuffer(mask_bytes, dtype=np.uint8)
    mask = cv2.imdecode(mask_array, cv2.IMREAD_UNCHANGED)
    
    return joints_uv_z, mask


def convert_uvz_to_xyz(joints_uv_z, depth_image, fx=475.0, fy=475.0, cx=160.0, cy=120.0):
    """
    Convert joint positions from (u, v, z) to (x, y, z) in camera coordinates.
    
    Args:
        joints_uv_z: (21, 3) joint positions in (u, v, z) format
            u, v: pixel coordinates
            z: depth in mm
        depth_image: original depth image for reference
        fx, fy: focal lengths
        cx, cy: principal point
        
    Returns:
        joints_xyz: (21, 3) joint positions in camera coordinates (mm)
    """
    joints_xyz = np.zeros((21, 3))
    
    for i, (u, v, z) in enumerate(joints_uv_z):
        # Convert pixel coordinates to camera coordinates
        # x = (u - cx) * z / fx
        # y = (v - cy) * z / fy
        x = (v - cx) * z / fx  # v is horizontal (column)
        y = (u - cy) * z / fy  # u is vertical (row)
        
        joints_xyz[i] = [x, y, z]
    
    return joints_xyz


def generate_hand_mesh(depth_image_path, output_path, mano_path, server_url="http://127.0.0.1:8080"):
    """
    Complete pipeline to generate hand mesh from depth image.
    """
    print("=" * 60)
    print("Hand Mesh Generation Pipeline")
    print("=" * 60)
    
    # Step 1: Load MANO model
    print("\n[Step 1] Loading MANO model...")
    mano = MANOModel(mano_path)
    
    # Use calibrated shape parameters (from the C++ code)
    shape_params = np.array([
        -2.61435056, -1.16743336, -2.80988378, 0.12670897, -0.08323125,
        2.28185672, -0.05833138, -2.95105206, -3.43976417, 0.30667237
    ])
    mano.init_rest_model(shape_params)
    
    # Step 2: Get joint positions from inference server
    print("\n[Step 2] Getting joint positions from inference server...")
    try:
        joints_uv_z, mask = get_joints_from_server(depth_image_path, server_url)
    except Exception as e:
        print(f"Error connecting to server: {e}")
        print("Make sure inference_server.py is running on port 8080")
        return None
    
    print(f"  Got {len(joints_uv_z)} joints")
    
    # Step 3: Convert UV+Z to XYZ coordinates
    print("\n[Step 3] Converting to 3D coordinates...")
    depth_img = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)
    joints_xyz = convert_uvz_to_xyz(joints_uv_z, depth_img)
    
    print("  Joint positions (xyz in mm):")
    joint_names = ["Wrist", "Thumb_CMC", "Thumb_MCP", "Thumb_IP", "Thumb_Tip",
                   "Index_MCP", "Index_PIP", "Index_DIP", "Index_Tip",
                   "Middle_MCP", "Middle_PIP", "Middle_DIP", "Middle_Tip",
                   "Ring_MCP", "Ring_PIP", "Ring_DIP", "Ring_Tip",
                   "Pinky_MCP", "Pinky_PIP", "Pinky_DIP", "Pinky_Tip"]
    
    for i, (name, xyz) in enumerate(zip(joint_names, joints_xyz)):
        print(f"    {i:2d}. {name:12s}: ({xyz[0]:7.2f}, {xyz[1]:7.2f}, {xyz[2]:7.2f})")
    
    # Step 4: Fit MANO model to joints
    print("\n[Step 4] Fitting MANO model to joint positions...")
    pose_pca, global_R, global_T, error = mano.fit_to_joints(joints_xyz, max_iter=200)
    
    print(f"  Fitting error: {error:.4f}")
    print(f"  Pose PCA: {pose_pca[:5]}...")
    print(f"  Global rotation: {global_R}")
    print(f"  Global translation: {global_T}")
    
    # Step 5: Generate mesh
    print("\n[Step 5] Generating hand mesh...")
    vertices = mano.get_posed_model(pose_pca, global_R, global_T)
    
    # Step 6: Export mesh
    print(f"\n[Step 6] Exporting mesh to {output_path}...")
    mano.export_obj(output_path, vertices)
    
    # Also save the mask
    mask_path = output_path.replace('.obj', '_mask.png')
    cv2.imwrite(mask_path, mask)
    print(f"  Saved segmentation mask to {mask_path}")
    
    # Export joints as OBJ points for visualization
    joints_path = output_path.replace('.obj', '_joints.obj')
    with open(joints_path, 'w') as f:
        f.write("# 21 Hand Joints\n")
        for i, (name, xyz) in enumerate(zip(joint_names, joints_xyz)):
            f.write(f"v {xyz[0]:.4f} {xyz[1]:.4f} {xyz[2]:.4f}\n")
    print(f"  Saved joint positions to {joints_path}")
    
    print("\n" + "=" * 60)
    print("Done! Generated files:")
    print(f"  - {output_path} (3D hand mesh)")
    print(f"  - {joints_path} (joint positions)")
    print(f"  - {mask_path} (segmentation mask)")
    print("=" * 60)
    
    return vertices, joints_xyz


def main():
    parser = argparse.ArgumentParser(description='Generate 3D hand mesh from depth image')
    parser.add_argument('--depth_image', type=str, default='org_depth_img_init.png',
                        help='Path to depth image')
    parser.add_argument('--output', type=str, default='hand_mesh.obj',
                        help='Output OBJ file path')
    parser.add_argument('--mano_path', type=str, 
                        default='../../InteractionReconstruction/InteractionReconstruction/data/mano/mano_r.json',
                        help='Path to MANO model JSON')
    parser.add_argument('--server_url', type=str, default='http://127.0.0.1:8080',
                        help='Inference server URL')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.depth_image):
        print(f"Error: Depth image not found: {args.depth_image}")
        return
    
    if not os.path.exists(args.mano_path):
        print(f"Error: MANO model not found: {args.mano_path}")
        return
    
    generate_hand_mesh(args.depth_image, args.output, args.mano_path, args.server_url)


if __name__ == "__main__":
    main()

