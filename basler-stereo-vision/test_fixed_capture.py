#!/usr/bin/env python3
"""
Test script to demonstrate that the image grab errors have been fixed.
This is a simplified version without OpenCV display for easy testing.
"""

import sys
import os
import time
import logging

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.camera.basler_camera import BaslerCamera

def main():
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
    
    print("=== TESTING FIXED BASLER STEREO CAMERA CAPTURE ===")
    
    # Create output directory for test images
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    # First, list all available cameras
    print("Scanning for available Basler cameras...")
    available_cameras = BaslerCamera.list_available_cameras()
    
    if len(available_cameras) == 0:
        print("No Basler cameras found on the network!")
        return
    elif len(available_cameras) < 2:
        print(f"Only {len(available_cameras)} camera found, but 2 cameras are required for stereo vision!")
        return
    else:
        print("Found exactly 2 cameras - perfect for stereo vision!")
    
    # Display the cameras that will be used
    for i in range(min(2, len(available_cameras))):
        cam = available_cameras[i]
        print(f"  Camera {i+1}: {cam['model_name']} "
              f"(S/N: {cam['serial_number']}, IP: {cam['ip_address']})")
    
    # Create camera instances for stereo setup
    camera_left = BaslerCamera(camera_index=0)   # First camera as left
    camera_right = BaslerCamera(camera_index=1)  # Second camera as right
    
    try:
        # Initialize both cameras
        print("\nInitializing left camera (Camera 1)...")
        if not camera_left.initialize():
            print("Failed to initialize left camera")
            return
            
        print("Initializing right camera (Camera 2)...")
        if not camera_right.initialize():
            print("Failed to initialize right camera")
            camera_left.close()  # Clean up left camera
            return
        
        print("\nStereo Camera Setup Complete!")
        print("Testing stereo frame capture (5 frames)...")
        
        success_count = 0
        total_frames = 5
        
        for i in range(total_frames):
            print(f"\nCapturing frame pair {i+1}/{total_frames}...")
            
            start_time = time.time()
            # Use the improved stereo capture method
            frame_left, frame_right = BaslerCamera.capture_stereo_frames(
                camera_left, camera_right, timeout_ms=3000
            )
            capture_time = (time.time() - start_time) * 1000
            
            if frame_left is not None and frame_right is not None:
                success_count += 1
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                print(f"✓ SUCCESS [{timestamp}] Frame pair {i+1} captured in {capture_time:.1f}ms")
                print(f"  Left frame: {frame_left.shape}")
                print(f"  Right frame: {frame_right.shape}")
                
                # Save frames for verification (optional) - now saves to output folder
                # import cv2
                # cv2.imwrite(os.path.join(output_dir, f"test_left_{i+1}.jpg"), frame_left)
                # cv2.imwrite(os.path.join(output_dir, f"test_right_{i+1}.jpg"), frame_right)
                
            else:
                print(f"✗ FAILED to capture frame pair {i+1}")
            
            time.sleep(1)  # Wait 1 second between captures
        
        print(f"\n=== CAPTURE TEST RESULTS ===")
        print(f"Successful captures: {success_count}/{total_frames}")
        print(f"Success rate: {(success_count/total_frames)*100:.1f}%")
        
        if success_count == total_frames:
            print("🎉 ALL TESTS PASSED - Image grab errors have been RESOLVED!")
        elif success_count > 0:
            print("⚠️  PARTIAL SUCCESS - Some captures worked, may need further tuning")
        else:
            print("❌ ALL TESTS FAILED - Further investigation needed")
                
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        
    finally:
        # Clean up both cameras
        print("\nClosing cameras...")
        camera_left.close()
        camera_right.close()
        print("Done!")

if __name__ == "__main__":
    main()
