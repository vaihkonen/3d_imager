#!/usr/bin/env python3
"""Final verification test for both cameras in stereo mode"""

from src.camera.basler_camera import BaslerCamera
import logging

logging.basicConfig(level=logging.WARNING)

print('='*60)
print('FINAL VERIFICATION TEST - Both Cameras Stereo Capture')
print('='*60)

# Initialize both cameras
print('\nInitializing cameras...')
cam_left = BaslerCamera(camera_index=0)
cam_right = BaslerCamera(camera_index=1)

left_ok = cam_left.initialize()
right_ok = cam_right.initialize()

if not left_ok or not right_ok:
    print(f'FAILED: left={left_ok}, right={right_ok}')
    exit(1)

print('✓ Both cameras initialized')

# Capture stereo pair
print('\nCapturing stereo frame pair...')
left_frame, right_frame = BaslerCamera.capture_stereo_frames(cam_left, cam_right, timeout_ms=15000)

if left_frame is not None and right_frame is not None:
    print(f'✓ SUCCESS! Captured stereo pair')
    print(f'  Left:  {left_frame.shape}, {left_frame.nbytes/1024/1024:.2f} MB')
    print(f'  Right: {right_frame.shape}, {right_frame.nbytes/1024/1024:.2f} MB')
else:
    print(f'✗ FAILED: left={left_frame is not None}, right={right_frame is not None}')

# Cleanup
cam_left.close()
cam_right.close()

print('\n' + '='*60)
print('VERIFICATION COMPLETE')
print('='*60)

