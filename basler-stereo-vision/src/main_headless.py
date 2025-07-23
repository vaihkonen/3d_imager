#!/usr/bin/env python3
"""
Headless version of main.py for testing the complete stereo vision pipeline
without requiring GUI display.
"""

import json
import cv2
import os
import numpy as np
from camera.basler_camera import BaslerCamera
from stereo.depth_estimation import estimate_depth, estimate_depth_sgbm
from utils.image_processing import preprocess_image

def main():
    print("=== BASLER STEREO VISION PIPELINE TEST ===")
    
    # Load camera configuration (adjust path relative to the script location)
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'camera_config.json')
    with open(config_path) as config_file:
        config = json.load(config_file)

    # Initialize cameras
    try:
        cameras_config = config['cameras']
        if len(cameras_config) < 2:
            print(f"Configuration must contain at least 2 cameras, found {len(cameras_config)}")
            return
        
        camera1_config = cameras_config[0]  # First camera
        camera2_config = cameras_config[1]  # Second camera
        
        print(f"Configuring cameras:")
        print(f"  Camera 1: {camera1_config['name']} at IP {camera1_config['ip_address']}")
        print(f"  Camera 2: {camera2_config['name']} at IP {camera2_config['ip_address']}")
        
    except KeyError as e:
        print(f"Missing camera configuration: {e}")
        return

    # Create camera instances using IP addresses from config
    camera1 = BaslerCamera(camera_ip=camera1_config['ip_address'])
    camera2 = BaslerCamera(camera_ip=camera2_config['ip_address'])

    # Initialize cameras
    print("\\nInitializing cameras...")
    if not camera1.initialize():
        print("Failed to initialize camera 1")
        return
        
    if not camera2.initialize():
        print("Failed to initialize camera 2")
        camera1.close()
        return

    print("Cameras initialized successfully!")

    try:
        # Capture frames using the improved stereo capture method
        print("\\nCapturing stereo frame pair...")
        frame1, frame2 = BaslerCamera.capture_stereo_frames(camera1, camera2, timeout_ms=5000)

        if frame1 is None or frame2 is None:
            print("Failed to capture frames from one or both cameras.")
            return
        
        print(f"Successfully captured stereo frames:")
        print(f"  Frame 1: {frame1.shape}")
        print(f"  Frame 2: {frame2.shape}")

        # Preprocess images
        print("\\nPreprocessing images...")
        processed_frame1 = preprocess_image(frame1)
        processed_frame2 = preprocess_image(frame2)
        
        print(f"Preprocessed frames:")
        print(f"  Processed frame 1: {processed_frame1.shape}")
        print(f"  Processed frame 2: {processed_frame2.shape}")

        # Estimate depth using both methods for comparison
        print("\\nComputing depth map using Block Matching...")
        disparity_bm, points_3d_bm = estimate_depth(processed_frame1, processed_frame2, debug=True)
        
        print("\\nComputing depth map using SGBM...")
        disparity_sgbm, points_3d_sgbm = estimate_depth_sgbm(processed_frame1, processed_frame2, debug=True)
        
        print(f"\\nDepth computation completed:")
        print(f"  BM Disparity map: {disparity_bm.shape}, range: {np.min(disparity_bm):.1f} to {np.max(disparity_bm):.1f}")
        print(f"  SGBM Disparity map: {disparity_sgbm.shape}, range: {np.min(disparity_sgbm):.1f} to {np.max(disparity_sgbm):.1f}")

        # Save output images for verification
        print("\\nSaving output images...")
        cv2.imwrite('output_frame1.jpg', frame1)
        cv2.imwrite('output_frame2.jpg', frame2) 
        
        # Save both disparity maps
        cv2.imwrite('output_disparity_bm.jpg', disparity_bm)
        cv2.imwrite('output_disparity_sgbm.jpg', disparity_sgbm)
        
        # Also save colored versions for better visualization
        disparity_bm_colored = cv2.applyColorMap(disparity_bm, cv2.COLORMAP_JET)
        disparity_sgbm_colored = cv2.applyColorMap(disparity_sgbm, cv2.COLORMAP_JET)
        cv2.imwrite('output_disparity_bm_colored.jpg', disparity_bm_colored)
        cv2.imwrite('output_disparity_sgbm_colored.jpg', disparity_sgbm_colored)
        
        print("Images saved:")
        print("  output_frame1.jpg - Raw left camera frame")
        print("  output_frame2.jpg - Raw right camera frame")
        print("  output_disparity_bm.jpg - Block Matching disparity (grayscale)")
        print("  output_disparity_sgbm.jpg - SGBM disparity (grayscale)")
        print("  output_disparity_bm_colored.jpg - Block Matching disparity (colored)")
        print("  output_disparity_sgbm_colored.jpg - SGBM disparity (colored)")
        print("\\n  Compare the different methods to see which works better for your setup!")
        
        print("\\n🎉 STEREO VISION PIPELINE COMPLETED SUCCESSFULLY!")
        
    except Exception as e:
        print(f"\\nError during processing: {str(e)}")
        
    finally:
        # Clean up both cameras
        print("\\nClosing cameras...")
        camera1.close()
        camera2.close()
        print("Pipeline test completed!")

if __name__ == "__main__":
    main()
