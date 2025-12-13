#!/usr/bin/env python3
"""
Test script for the LSTMPose inference server.
Sends sample pose data and displays the refined output.
"""

import requests
import numpy as np
import math

SERVER_URL = "http://127.0.0.1:8081"

def test_lstm_pose():
    print(f"Testing LSTMPose server at {SERVER_URL}")
    print("=" * 50)
    
    # The model expects 22 features (joint angles in radians, range -pi to pi)
    # Let's create a sequence of poses to test temporal smoothing
    
    np.random.seed(42)
    
    # Simulate a base pose with some noise (22 joint angles)
    base_pose = np.random.uniform(-0.5, 0.5, 22).astype(np.float32)
    
    print("\nSending 5 frames to test temporal smoothing...\n")
    
    for frame_id in range(5):
        # Add some noise to simulate frame-to-frame jitter
        noise = np.random.normal(0, 0.05, 22).astype(np.float32)
        noisy_pose = base_pose + noise
        
        # Clip to valid range
        noisy_pose = np.clip(noisy_pose, -math.pi, math.pi)
        
        # Convert pose to space-separated string
        pose_str = ' '.join([str(p) for p in noisy_pose])
        
        # Send request
        files = {
            'pose': ('pose', pose_str.encode(), 'text/plain'),
            'id': ('id', b'1', 'text/plain')  # Pipeline ID for tracking state
        }
        
        response = requests.post(SERVER_URL, files=files)
        
        if response.status_code != 200:
            print(f"Frame {frame_id}: Error - status {response.status_code}")
            continue
        
        # Parse response
        refined_pose = [float(x) for x in response.text.split()]
        refined_pose = np.array(refined_pose)
        
        # Calculate how much the LSTM changed the input
        diff = np.abs(refined_pose - noisy_pose)
        
        print(f"Frame {frame_id}:")
        print(f"  Input noise std:  {noise.std():.4f}")
        print(f"  Correction magnitude (mean): {diff.mean():.4f}")
        print(f"  Correction magnitude (max):  {diff.max():.4f}")
    
    print("\n" + "=" * 50)
    print("LSTM Temporal Smoothing Test Complete!")
    print("=" * 50)
    
    # Show sample of final refined pose
    print(f"\nSample refined values (first 5 of 22 angles):")
    for i in range(5):
        print(f"  Joint {i}: {refined_pose[i]:.4f} rad ({math.degrees(refined_pose[i]):.1f}°)")
    
    print("\nThe LSTM maintains internal state across frames,")
    print("smoothing out noise and temporal jitter in the pose estimates.")

if __name__ == "__main__":
    test_lstm_pose()

