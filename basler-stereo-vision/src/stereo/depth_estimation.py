import cv2
import numpy as np

def estimate_depth(frame_left, frame_right):
    # Convert frames to grayscale if they aren't already
    if len(frame_left.shape) == 3:  # Color image
        gray_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
    else:  # Already grayscale
        gray_left = frame_left
        
    if len(frame_right.shape) == 3:  # Color image
        gray_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)
    else:  # Already grayscale
        gray_right = frame_right

    # Create a stereo block matcher
    stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)

    # Compute the disparity map
    disparity = stereo.compute(gray_left, gray_right)

    # Normalize the disparity map for visualization
    disparity_normalized = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)

    # Create a 3D image from the disparity map
    h, w = gray_left.shape
    f = 0.8 * w  # Focal length
    Q = np.float32([[1, 0, 0, -0.5 * w],
                     [0, -1, 0, 0.5 * h],  # Flip the y-axis
                     [0, 0, 0, -f],
                     [0, 0, 1, 0]])

    # Reproject the disparity map to 3D
    points_3D = cv2.reprojectImageTo3D(disparity, Q)

    return disparity_normalized, points_3D 