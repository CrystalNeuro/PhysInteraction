#!/usr/bin/env python3
"""
Test script for the inference server.
Sends the sample depth image and displays the results.
"""

import requests
import base64
import json
import numpy as np
import cv2

SERVER_URL = "http://127.0.0.1:8080"
DEPTH_IMAGE_PATH = "org_depth_img_init.png"

def test_inference():
    print(f"Testing inference server at {SERVER_URL}")
    print(f"Using depth image: {DEPTH_IMAGE_PATH}")
    
    # Prepare the request
    with open(DEPTH_IMAGE_PATH, 'rb') as f:
        files = {
            'depth': ('depth.png', f, 'image/png'),
            'begin': ('begin', b'1', 'text/plain')
        }
        
        print("\nSending request...")
        response = requests.post(SERVER_URL, files=files)
    
    if response.status_code != 200:
        print(f"Error: Server returned status {response.status_code}")
        return
    
    # Parse the response
    result = response.json()
    
    # Decode joints
    joints_bytes = base64.b64decode(result['joints'])
    joints = np.frombuffer(joints_bytes, dtype=np.float32).reshape(21, 3)
    
    # Decode segmentation mask
    mask_bytes = base64.b64decode(result['segment'])
    mask_array = np.frombuffer(mask_bytes, dtype=np.uint8)
    mask = cv2.imdecode(mask_array, cv2.IMREAD_UNCHANGED)
    
    print("\n" + "="*50)
    print("INFERENCE RESULTS")
    print("="*50)
    
    print("\n21 Hand Joint Positions (u, v, z):")
    print("-"*40)
    joint_names = [
        "Wrist",
        "Thumb_CMC", "Thumb_MCP", "Thumb_IP", "Thumb_Tip",
        "Index_MCP", "Index_PIP", "Index_DIP", "Index_Tip",
        "Middle_MCP", "Middle_PIP", "Middle_DIP", "Middle_Tip",
        "Ring_MCP", "Ring_PIP", "Ring_DIP", "Ring_Tip",
        "Pinky_MCP", "Pinky_PIP", "Pinky_DIP", "Pinky_Tip"
    ]
    
    for i, (name, joint) in enumerate(zip(joint_names, joints)):
        print(f"{i:2d}. {name:12s}: u={joint[0]:6.1f}, v={joint[1]:6.1f}, z={joint[2]:7.2f}mm")
    
    print(f"\nSegmentation mask shape: {mask.shape}")
    print(f"Mask value range: {mask.min()} - {mask.max()}")
    
    # Save visualization
    depth_img = cv2.imread(DEPTH_IMAGE_PATH, cv2.IMREAD_UNCHANGED)
    
    # Create visualization with joints overlaid
    if len(depth_img.shape) == 2:
        vis_img = cv2.cvtColor((depth_img / depth_img.max() * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    else:
        vis_img = depth_img.copy()
    
    # Draw joints (scale from 30x40 heatmap coords to 240x320 image)
    for i, joint in enumerate(joints):
        u, v = int(joint[0]), int(joint[1])
        if 0 <= u < vis_img.shape[0] and 0 <= v < vis_img.shape[1]:
            cv2.circle(vis_img, (v, u), 3, (0, 255, 0), -1)
            cv2.putText(vis_img, str(i), (v+5, u), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)
    
    # Save outputs
    cv2.imwrite("result_joints.png", vis_img)
    cv2.imwrite("result_mask.png", mask)
    
    print("\nSaved visualizations:")
    print("  - result_joints.png (depth image with joint positions)")
    print("  - result_mask.png (segmentation mask)")
    print("\nTest completed successfully!")

if __name__ == "__main__":
    test_inference()

