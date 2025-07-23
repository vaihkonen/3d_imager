import json
import cv2
import os
from camera.basler_camera import BaslerCamera
from stereo.depth_estimation import estimate_depth
from utils.image_processing import preprocess_image

def main():
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
    print("Initializing cameras...")
    if not camera1.initialize():
        print("Failed to initialize camera 1")
        return
        
    if not camera2.initialize():
        print("Failed to initialize camera 2")
        camera1.close()
        return

    print("Cameras initialized successfully!")

    # Capture frames using the improved stereo capture method
    print("Capturing stereo frame pair...")
    frame1, frame2 = BaslerCamera.capture_stereo_frames(camera1, camera2, timeout_ms=5000)

    if frame1 is None or frame2 is None:
        print("Failed to capture frames from one or both cameras.")
        camera1.close()
        camera2.close()
        return
    
    print(f"Successfully captured stereo frames:")
    print(f"  Frame 1: {frame1.shape}")
    print(f"  Frame 2: {frame2.shape}")

    # Preprocess images
    processed_frame1 = preprocess_image(frame1)
    processed_frame2 = preprocess_image(frame2)

    # Estimate depth
    print("Computing depth map...")
    disparity_map, points_3d = estimate_depth(processed_frame1, processed_frame2)
    
    print(f"Depth computation completed:")
    print(f"  Disparity map shape: {disparity_map.shape}")
    print(f"  3D points shape: {points_3d.shape}")

    # Display the depth map
    print("Displaying depth map (press any key to close)...")
    cv2.imshow('Disparity Map', disparity_map.astype('uint8'))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Clean up
    camera1.close()
    camera2.close()

if __name__ == "__main__":
    main()