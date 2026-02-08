#!/usr/bin/env python3
"""
Simple test to verify both cameras work sequentially
"""

from src.camera.basler_camera import BaslerCamera
import logging
import time

logging.basicConfig(level=logging.INFO)

print("="*60)
print("Testing Stereo Camera Setup")
print("="*60)

# Test Camera 1
print("\n[1/4] Initializing Camera 1 (Left)...")
cam1 = BaslerCamera(camera_index=0)
if not cam1.initialize():
    print("FAILED to initialize camera 1")
    exit(1)
print("✓ Camera 1 initialized")

# Test Camera 2
print("\n[2/4] Initializing Camera 2 (Right)...")
cam2 = BaslerCamera(camera_index=1)
if not cam2.initialize():
    print("FAILED to initialize camera 2")
    cam1.close()
    exit(1)
print("✓ Camera 2 initialized")

# Capture from Camera 1
print("\n[3/4] Capturing from Camera 1...")
frame1 = cam1.capture_frame(timeout_ms=15000)
if frame1 is None:
    print("FAILED to capture from camera 1")
else:
    print(f"✓ Camera 1 captured: {frame1.shape}, {frame1.nbytes/1024/1024:.2f} MB")

# Capture from Camera 2
print("\n[4/4] Capturing from Camera 2...")
frame2 = cam2.capture_frame(timeout_ms=15000)
if frame2 is None:
    print("FAILED to capture from camera 2")
else:
    print(f"✓ Camera 2 captured: {frame2.shape}, {frame2.nbytes/1024/1024:.2f} MB")

# Cleanup
print("\nClosing cameras...")
cam1.close()
cam2.close()

print("\n" + "="*60)
if frame1 is not None and frame2 is not None:
    print("SUCCESS! Both cameras working!")
else:
    print("PARTIAL SUCCESS - some cameras failed")
print("="*60)

