#!/usr/bin/env python3
"""
Enhanced stereo vision pipeline with rectification for better disparity maps.
"""

import json
import cv2
import os
import numpy as np
from camera.basler_camera import BaslerCamera
from stereo.depth_estimation import estimate_depth_sgbm
from stereo.rectification import align_stereo_pair_horizontally
from utils.image_processing import preprocess_image

def main():
    print("=== ENHANCED BASLER STEREO VISION PIPELINE ===")
    
    # Create output directory if it doesn't exist
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {os.path.abspath(output_dir)}")
    
    # Use auto-discovery instead of config file
    print("Auto-discovering Basler cameras...")
    
    # Create camera instances using auto-discovery (camera indices)
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
        # Capture frames
        print("\nCapturing stereo frame pair...")
        frame1, frame2 = BaslerCamera.capture_stereo_frames(camera1, camera2, timeout_ms=5000)

        if frame1 is None or frame2 is None:
            print("Failed to capture frames from one or both cameras.")
            return
        
        print(f"Successfully captured stereo frames:")
        print(f"  Frame 1: {frame1.shape}")
        print(f"  Frame 2: {frame2.shape}")

        # Save original frames
        cv2.imwrite(os.path.join(output_dir, 'frame1_original.jpg'), frame1)
        cv2.imwrite(os.path.join(output_dir, 'frame2_original.jpg'), frame2)

        # Apply horizontal alignment
        print("\nApplying stereo alignment...")
        frame1_aligned, frame2_aligned = align_stereo_pair_horizontally(frame1, frame2, debug=True)
        
        # Save aligned frames
        cv2.imwrite(os.path.join(output_dir, 'frame1_aligned.jpg'), frame1_aligned)
        cv2.imwrite(os.path.join(output_dir, 'frame2_aligned.jpg'), frame2_aligned)

        # Preprocess aligned images
        print("\nPreprocessing aligned images...")
        processed_frame1 = preprocess_image(frame1_aligned)
        processed_frame2 = preprocess_image(frame2_aligned)

        # Compute depth using SGBM (which worked better)
        print("\nComputing enhanced depth map...")
        disparity_map, points_3d = estimate_depth_sgbm(processed_frame1, processed_frame2, debug=True)
        
        print(f"\nDepth computation completed:")
        print(f"  Disparity map: {disparity_map.shape}, range: {np.min(disparity_map):.1f} to {np.max(disparity_map):.1f}")

        # Save results
        print("\nSaving enhanced output images...")
        
        # Save disparity maps
        cv2.imwrite(os.path.join(output_dir, 'disparity_enhanced.jpg'), disparity_map)
        disparity_colored = cv2.applyColorMap(disparity_map, cv2.COLORMAP_JET)
        cv2.imwrite(os.path.join(output_dir, 'disparity_enhanced_colored.jpg'), disparity_colored)
        
        # Create a side-by-side comparison
        comparison = np.hstack((frame1_aligned, frame2_aligned))
        cv2.imwrite(os.path.join(output_dir, 'stereo_pair.jpg'), comparison)
        
        print("Enhanced images saved to output folder:")
        print("  output/frame1_original.jpg - Original left frame")
        print("  output/frame2_original.jpg - Original right frame")
        print("  output/frame1_aligned.jpg - Aligned left frame")
        print("  output/frame2_aligned.jpg - Aligned right frame")
        print("  output/stereo_pair.jpg - Side-by-side stereo pair")
        print("  output/disparity_enhanced.jpg - Enhanced disparity map")
        print("  output/disparity_enhanced_colored.jpg - Enhanced colored disparity map")
        
        print("\n🎉 ENHANCED STEREO VISION PIPELINE COMPLETED!")
        print("\nTips for better results:")
        print("- Ensure cameras are mounted as parallel as possible")
        print("- Use good lighting and textured scenes")
        print("- Consider proper stereo calibration for production use")
        
    except Exception as e:
        print(f"\nError during processing: {str(e)}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Clean up
        print("\nClosing cameras...")
        camera1.close()
        camera2.close()
        print("Enhanced pipeline test completed!")

if __name__ == "__main__":
    main()
