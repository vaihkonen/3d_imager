import json
import cv2
import os
import numpy as np
from camera.basler_camera import BaslerCamera
from stereo.depth_estimation import estimate_depth
from utils.image_processing import preprocess_image

def main():
    print("=== BASLER STEREO VISION PIPELINE ===")
    
    # Create output directory for any future image saves
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Use auto-discovery instead of config file
    print("Auto-discovering Basler cameras...")
    
    # Create camera instances using auto-discovery
    camera1 = BaslerCamera(camera_index=0)  # First camera found
    camera2 = BaslerCamera(camera_index=1)  # Second camera found

    # Initialize cameras
    print("Initializing cameras...")
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
        print("Capturing stereo frame pair...")
        frame1, frame2 = BaslerCamera.capture_stereo_frames(camera1, camera2, timeout_ms=5000)

        if frame1 is None or frame2 is None:
            print("Failed to capture frames from one or both cameras.")
            return
        
        print(f"Successfully captured stereo frames:")
        print(f"  Frame 1: {frame1.shape}")
        print(f"  Frame 2: {frame2.shape}")

        # Save original frames to output folder
        cv2.imwrite(os.path.join(output_dir, 'frame1_original.jpg'), frame1)
        cv2.imwrite(os.path.join(output_dir, 'frame2_original.jpg'), frame2)
        print("Original frames saved to output folder")

        # Preprocess images
        print("Preprocessing images...")
        processed_frame1 = preprocess_image(frame1)
        processed_frame2 = preprocess_image(frame2)

        # Estimate depth
        print("Computing depth map...")
        disparity_map, points_3d = estimate_depth(processed_frame1, processed_frame2)
        
        print(f"Depth computation completed:")
        print(f"  Disparity map shape: {disparity_map.shape}")
        print(f"  3D points shape: {points_3d.shape}")

        # Save disparity map to output folder
        disparity_normalized = cv2.normalize(disparity_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        cv2.imwrite(os.path.join(output_dir, 'disparity_map.jpg'), disparity_normalized)
        
        # Create colored disparity map
        disparity_colored = cv2.applyColorMap(disparity_normalized, cv2.COLORMAP_JET)
        cv2.imwrite(os.path.join(output_dir, 'disparity_map_colored.jpg'), disparity_colored)
        
        print("Disparity maps saved to output folder")

        # Display the depth map
        print("Displaying depth map (press any key to close)...")
        cv2.imshow('Disparity Map', disparity_normalized)
        cv2.imshow('Disparity Map (Colored)', disparity_colored)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        print("\nFiles saved:")
        print("  output/frame1_original.jpg - Left camera frame")
        print("  output/frame2_original.jpg - Right camera frame") 
        print("  output/disparity_map.jpg - Grayscale disparity map")
        print("  output/disparity_map_colored.jpg - Colored disparity map")

    except Exception as e:
        print(f"Error during processing: {str(e)}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Clean up
        print("\nClosing cameras...")
        camera1.close()
        camera2.close()
        print("Pipeline completed!")

if __name__ == "__main__":
    main()