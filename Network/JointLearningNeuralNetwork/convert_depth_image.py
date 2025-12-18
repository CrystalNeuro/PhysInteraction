#!/usr/bin/env python3
"""
Script to convert depth images from various formats to the format expected by JointLearningNN.

The model expects:
- Size: 240×320 pixels
- Format: 16-bit PNG (uint16)
- Values: Depth in millimeters (0-65535 range)

Usage:
    python convert_depth_image.py --input dataset_depth.png --output converted_depth.png
"""

import argparse
import cv2
import numpy as np
import os

def convert_depth_image(input_path, output_path, target_size=(240, 320)):
    """
    Convert a depth image to the format expected by JointLearningNN.
    
    Args:
        input_path: Path to input depth image
        output_path: Path to save converted depth image
        target_size: Target size (height, width)
    """
    print(f"Loading: {input_path}")
    
    # Load image as 16-bit
    depth = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    
    if depth is None:
        raise ValueError(f"Could not load image: {input_path}")
    
    print(f"  Input shape: {depth.shape}")
    print(f"  Input dtype: {depth.dtype}")
    print(f"  Input range: [{depth.min()}, {depth.max()}]")
    
    # Convert to uint16 if needed
    if depth.dtype == np.uint8:
        print("  Converting from 8-bit to 16-bit...")
        depth = depth.astype(np.uint16) * 256
    elif depth.dtype == np.float32 or depth.dtype == np.float64:
        print("  Converting from float to uint16...")
        # Assume float is in meters, convert to mm
        depth = (depth * 1000).astype(np.uint16)
    
    # Resize if needed
    if depth.shape != target_size:
        print(f"  Resizing from {depth.shape} to {target_size}...")
        # Use INTER_NEAREST to preserve depth values (no interpolation artifacts)
        depth_resized = cv2.resize(depth, (target_size[1], target_size[0]), 
                                   interpolation=cv2.INTER_NEAREST)
    else:
        depth_resized = depth
    
    # Save as 16-bit PNG
    print(f"Saving: {output_path}")
    cv2.imwrite(output_path, depth_resized)
    
    # Verify
    verify = cv2.imread(output_path, cv2.IMREAD_UNCHANGED)
    print(f"  Output shape: {verify.shape}")
    print(f"  Output dtype: {verify.dtype}")
    print(f"  Output range: [{verify.min()}, {verify.max()}]")
    print(f"  Non-zero pixels: {np.count_nonzero(verify)} / {verify.size} ({100*np.count_nonzero(verify)/verify.size:.1f}%)")
    
    return depth_resized

def main():
    parser = argparse.ArgumentParser(description="Convert depth images to JointLearningNN format.")
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input depth image')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to output depth image (default: input_converted.png)')
    parser.add_argument('--height', type=int, default=240,
                        help='Target height (default: 240)')
    parser.add_argument('--width', type=int, default=320,
                        help='Target width (default: 320)')
    args = parser.parse_args()
    
    if args.output is None:
        base, ext = os.path.splitext(args.input)
        args.output = f"{base}_converted{ext}"
    
    convert_depth_image(args.input, args.output, target_size=(args.height, args.width))
    
    print(f"\n✅ Conversion complete!")
    print(f"You can now use: python generate_hand_mesh.py --depth_image {args.output}")

if __name__ == '__main__':
    main()
