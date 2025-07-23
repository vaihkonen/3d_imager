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
    
    # Load camera configuration
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'camera_config.json')
    with open(config_path) as config_file:
        config = json.load(config_file)

    # Initialize cameras
    try:
        cameras_config = config['cameras']
        if len(cameras_config) < 2:
            print(f"Configuration must contain at least 2 cameras, found {len(cameras_config)}")
            return
        
        camera1_config = cameras_config[0]
        camera2_config = cameras_config[1]
        
        print(f"Configuring cameras:")
        print(f"  Camera 1: {camera1_config['name']} at IP {camera1_config['ip_address']}")
        print(f"  Camera 2: {camera2_config['name']} at IP {camera2_config['ip_address']}")
        
    except KeyError as e:
        print(f"Missing camera configuration: {e}")
        return

    # Create camera instances
    camera1 = BaslerCamera(camera_ip=camera1_config['ip_address'])
    camera2 = BaslerCamera(camera_ip=camera2_config['ip_address'])

    # Initialize cameras
    print("\\nInitializing cameras...")
    if not camera1.initialize() or not camera2.initialize():
        print("Failed to initialize cameras")
        return

    print("Cameras initialized successfully!")

    try:
        # Capture frames
        print("\\nCapturing stereo frame pair...")
        frame1, frame2 = BaslerCamera.capture_stereo_frames(camera1, camera2, timeout_ms=5000)

        if frame1 is None or frame2 is None:
            print("Failed to capture frames from one or both cameras.")
            return
        
        print(f"Successfully captured stereo frames:")
        print(f"  Frame 1: {frame1.shape}")
        print(f"  Frame 2: {frame2.shape}")

        # Save original frames
        cv2.imwrite('output_frame1_original.jpg', frame1)
        cv2.imwrite('output_frame2_original.jpg', frame2)

        # Apply horizontal alignment
        print("\\nApplying stereo alignment...")
        frame1_aligned, frame2_aligned = align_stereo_pair_horizontally(frame1, frame2, debug=True)
        
        # Save aligned frames
        cv2.imwrite('output_frame1_aligned.jpg', frame1_aligned)
        cv2.imwrite('output_frame2_aligned.jpg', frame2_aligned)

        # Preprocess aligned images
        print("\\nPreprocessing aligned images...")
        processed_frame1 = preprocess_image(frame1_aligned)
        processed_frame2 = preprocess_image(frame2_aligned)

        # Compute depth using SGBM (which worked better)
        print("\\nComputing enhanced depth map...")
        disparity_map, points_3d = estimate_depth_sgbm(processed_frame1, processed_frame2, debug=True)
        
        print(f"\\nDepth computation completed:")
        print(f"  Disparity map: {disparity_map.shape}, range: {np.min(disparity_map):.1f} to {np.max(disparity_map):.1f}")

        # Save results
        print("\\nSaving enhanced output images...")
        
        # Save disparity maps
        cv2.imwrite('output_disparity_enhanced.jpg', disparity_map)
        disparity_colored = cv2.applyColorMap(disparity_map, cv2.COLORMAP_JET)
        cv2.imwrite('output_disparity_enhanced_colored.jpg', disparity_colored)
        
        # Create a side-by-side comparison
        comparison = np.hstack((frame1_aligned, frame2_aligned))
        cv2.imwrite('output_stereo_pair.jpg', comparison)
        
        print("Enhanced images saved:")
        print("  output_frame1_original.jpg - Original left frame")
        print("  output_frame2_original.jpg - Original right frame")
        print("  output_frame1_aligned.jpg - Aligned left frame")
        print("  output_frame2_aligned.jpg - Aligned right frame")
        print("  output_stereo_pair.jpg - Side-by-side stereo pair")
        print("  output_disparity_enhanced.jpg - Enhanced disparity map")
        print("  output_disparity_enhanced_colored.jpg - Enhanced colored disparity map")
        
        print("\\n🎉 ENHANCED STEREO VISION PIPELINE COMPLETED!")
        print("\\nTips for better results:")
        print("- Ensure cameras are mounted as parallel as possible")
        print("- Use good lighting and textured scenes")
        print("- Consider proper stereo calibration for production use")
        
    except Exception as e:
        print(f"\\nError during processing: {str(e)}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Clean up
        print("\\nClosing cameras...")
        camera1.close()
        camera2.close()
        print("Enhanced pipeline test completed!")

if __name__ == "__main__":
    main()
