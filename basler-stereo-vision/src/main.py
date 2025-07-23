import json
import cv2
from camera.basler_camera import BaslerCamera
from stereo.depth_estimation import estimate_depth
from utils.image_processing import preprocess_image

def main():
    # Load camera configuration
    with open('config/camera_config.json') as config_file:
        config = json.load(config_file)

    # Initialize cameras
    camera1 = BaslerCamera(config['camera1'])
    camera2 = BaslerCamera(config['camera2'])

    camera1.initialize()
    camera2.initialize()

    # Capture frames
    frame1 = camera1.capture_frame()
    frame2 = camera2.capture_frame()

    # Preprocess images
    processed_frame1 = preprocess_image(frame1)
    processed_frame2 = preprocess_image(frame2)

    # Estimate depth
    depth_map = estimate_depth(processed_frame1, processed_frame2)

    # Display or save the depth map as needed
    cv2.imshow('Depth Map', depth_map)
    cv2.waitKey(0)

    # Clean up
    camera1.close()
    camera2.close()

if __name__ == "__main__":
    main()