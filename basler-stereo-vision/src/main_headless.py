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
    
    # Create output directory for captured images
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {os.path.abspath(output_dir)}")
    
    # Use auto-discovery instead of config file
    print("Auto-discovering Basler cameras...")
    
    # Create camera instances using auto-discovery
    camera1 = BaslerCamera(camera_index=0)  # First camera found
    camera2 = BaslerCamera(camera_index=1)  # Second camera found

    # Initialize cameras
    print("\nInitializing cameras...")
    if not camera1.initialize():
        print("Failed to initialize camera 1 (index 0)")
        return
        
    if not camera2.initialize():
        print("Failed to initialize camera 2 (index 1)")
        camera1.close()
        return

    print("Cameras initialized successfully!")

    try:
        # Capture frames using the improved stereo capture method
        print("\nCapturing stereo frame pair...")
        frame1, frame2 = BaslerCamera.capture_stereo_frames(camera1, camera2, timeout_ms=5000)

        if frame1 is None or frame2 is None:
            print("Failed to capture frames from one or both cameras.")
            return
        
        print(f"Successfully captured stereo frames:")
        print(f"  Frame 1: {frame1.shape}")
        print(f"  Frame 2: {frame2.shape}")

        # Preprocess images
        print("\nPreprocessing images...")
        processed_frame1 = preprocess_image(frame1)
        processed_frame2 = preprocess_image(frame2)
        
        print(f"Preprocessed frames:")
        print(f"  Processed frame 1: {processed_frame1.shape}")
        print(f"  Processed frame 2: {processed_frame2.shape}")

        # Estimate depth using both methods for comparison
        print("\nComputing depth map using Block Matching...")
        disparity_bm, points_3d_bm = estimate_depth(processed_frame1, processed_frame2, debug=True)
        
        print("\nComputing depth map using SGBM...")
        disparity_sgbm, points_3d_sgbm = estimate_depth_sgbm(processed_frame1, processed_frame2, debug=True)
        
        print(f"\nDepth computation completed:")
        print(f"  BM Disparity map: {disparity_bm.shape}, range: {np.min(disparity_bm):.1f} to {np.max(disparity_bm):.1f}")
        print(f"  SGBM Disparity map: {disparity_sgbm.shape}, range: {np.min(disparity_sgbm):.1f} to {np.max(disparity_sgbm):.1f}")

        # Save output images for verification to output folder
        print("\nSaving output images...")
        cv2.imwrite(os.path.join(output_dir, 'frame1.jpg'), frame1)
        cv2.imwrite(os.path.join(output_dir, 'frame2.jpg'), frame2) 
        
        # Save both disparity maps
        cv2.imwrite(os.path.join(output_dir, 'disparity_bm.jpg'), disparity_bm)
        cv2.imwrite(os.path.join(output_dir, 'disparity_sgbm.jpg'), disparity_sgbm)
        
        # Also save colored versions for better visualization
        disparity_bm_colored = cv2.applyColorMap(disparity_bm, cv2.COLORMAP_JET)
        disparity_sgbm_colored = cv2.applyColorMap(disparity_sgbm, cv2.COLORMAP_JET)
        cv2.imwrite(os.path.join(output_dir, 'disparity_bm_colored.jpg'), disparity_bm_colored)
        cv2.imwrite(os.path.join(output_dir, 'disparity_sgbm_colored.jpg'), disparity_sgbm_colored)
        
        print("Images saved to output folder:")
        print("  output/frame1.jpg - Raw left camera frame")
        print("  output/frame2.jpg - Raw right camera frame")
        print("  output/disparity_bm.jpg - Block Matching disparity (grayscale)")
        print("  output/disparity_sgbm.jpg - SGBM disparity (grayscale)")
        print("  output/disparity_bm_colored.jpg - Block Matching disparity (colored)")
        print("  output/disparity_sgbm_colored.jpg - SGBM disparity (colored)")
        print("\n  Compare the different methods to see which works better for your setup!")
        
        print("\n🎉 STEREO VISION PIPELINE COMPLETED SUCCESSFULLY!")
        
    except Exception as e:
        print(f"\nError during processing: {str(e)}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Clean up both cameras
        print("\nClosing cameras...")
        camera1.close()
        camera2.close()
        print("Pipeline test completed!")

if __name__ == "__main__":
    main()
