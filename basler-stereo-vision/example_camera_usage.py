#!/usr/bin/env python3
"""
Example usage of the BaslerCamera class with pypylon integration.
"""

import sys
import os
import time
import cv2
import logging

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.camera.basler_camera import BaslerCamera

def main():
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # First, list all available cameras
    print("Scanning for available Basler cameras...")
    available_cameras = BaslerCamera.list_available_cameras()
    
    if len(available_cameras) == 0:
        print("No Basler cameras found on the network!")
        return
    elif len(available_cameras) < 2:
        print(f"Only {len(available_cameras)} camera found, but 2 cameras are required for stereo vision!")
        print("Available camera:")
        for cam in available_cameras:
            print(f"  Index {cam['index']}: {cam['model_name']} "
                  f"(S/N: {cam['serial_number']}, IP: {cam['ip_address']})")
        return
    elif len(available_cameras) > 2:
        print(f"Found {len(available_cameras)} cameras. Using first 2 for stereo vision:")
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
        
        # Get camera info for both cameras
        print("\nStereo Camera Setup Complete!")
        
        info_left = camera_left.get_camera_info()
        if info_left:
            print("\nLeft Camera Information:")
            for key, value in info_left.items():
                print(f"  {key}: {value}")
        
        info_right = camera_right.get_camera_info()
        if info_right:
            print("\nRight Camera Information:")
            for key, value in info_right.items():
                print(f"  {key}: {value}")
        
        print("\nStarting stereo frame capture every 15 seconds (press 'q' to quit)...")
        
        # Capture frames in a loop - once every 15 seconds from both cameras
        frame_count = 0
        last_capture_time = 0
        capture_interval = 15  # seconds
        
        while True:
            current_time = time.time()
            
            # Check if it's time to capture a new frame
            if current_time - last_capture_time >= capture_interval:
                # Capture frames from both cameras simultaneously
                frame_left = camera_left.capture_frame()
                frame_right = camera_right.capture_frame()
                
                if frame_left is not None and frame_right is not None:
                    frame_count += 1
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                    print(f"\n[{timestamp}] Captured stereo frame pair {frame_count}")
                    print(f"  Left frame shape: {frame_left.shape}")
                    print(f"  Right frame shape: {frame_right.shape}")
                    
                    # Display both frames
                    cv2.imshow('Left Camera Feed', frame_left)
                    cv2.imshow('Right Camera Feed', frame_right)
                    
                    # Save frames with timestamp
                    filename_left = f"frame_left_{frame_count}_{int(current_time)}.jpg"
                    filename_right = f"frame_right_{frame_count}_{int(current_time)}.jpg"
                    
                    cv2.imwrite(filename_left, frame_left)
                    cv2.imwrite(filename_right, frame_right)
                    
                    print(f"  Saved left frame as: {filename_left}")
                    print(f"  Saved right frame as: {filename_right}")
                    
                    last_capture_time = current_time
                    
                elif frame_left is None:
                    print("Failed to capture frame from left camera")
                elif frame_right is None:
                    print("Failed to capture frame from right camera")
                else:
                    print("Failed to capture frames from both cameras")
            
            # Check for quit key (check every 100ms)
            key = cv2.waitKey(100) & 0xFF
            if key == ord('q'):
                break
            
            # Show countdown to next capture
            time_until_next = capture_interval - (current_time - last_capture_time)
            if time_until_next > 0:
                print(f"\rNext capture in: {time_until_next:.1f} seconds", end="", flush=True)
                
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        
    finally:
        # Clean up both cameras
        print("\nClosing cameras...")
        camera_left.close()
        camera_right.close()
        cv2.destroyAllWindows()
        print("Done!")

if __name__ == "__main__":
    main()
