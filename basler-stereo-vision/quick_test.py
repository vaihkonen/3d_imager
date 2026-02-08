#!/usr/bin/env python3
"""Quick test - just open and close cameras"""
import sys
sys.path.insert(0, '/Users/juhanivaihkonen/PycharmProjects/3d_imager/basler-stereo-vision')

from src.camera.basler_camera import BaslerCamera
import logging

logging.basicConfig(level=logging.WARNING)  # Reduce noise

print("Test 1: Camera 0 only")
try:
    cam = BaslerCamera(camera_index=0)
    if cam.initialize():
        print("  ✓ Camera 0 init OK")
        frame = cam.capture_frame(timeout_ms=15000)
        if frame is not None:
            print(f"  ✓ Camera 0 capture OK: {frame.shape}")
        else:
            print("  ✗ Camera 0 capture FAILED")
        cam.close()
        print("  ✓ Camera 0 closed")
    else:
        print("  ✗ Camera 0 init FAILED")
except Exception as e:
    print(f"  ✗ Camera 0 ERROR: {e}")

print("\nTest 2: Camera 1 only")
try:
    cam = BaslerCamera(camera_index=1)
    if cam.initialize():
        print("  ✓ Camera 1 init OK")
        frame = cam.capture_frame(timeout_ms=15000)
        if frame is not None:
            print(f"  ✓ Camera 1 capture OK: {frame.shape}")
        else:
            print("  ✗ Camera 1 capture FAILED")
        cam.close()
        print("  ✓ Camera 1 closed")
    else:
        print("  ✗ Camera 1 init FAILED")
except Exception as e:
    print(f"  ✗ Camera 1 ERROR: {e}")

print("\nDone!")

