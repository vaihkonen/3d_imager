import cv2
import numpy as np

def estimate_depth(frame_left, frame_right, debug=False):
    """
    Estimate depth from stereo image pair with improved parameters for high-resolution images.
    
    Args:
        frame_left: Left camera image
        frame_right: Right camera image  
        debug: If True, print debug information
        
    Returns:
        tuple: (disparity_normalized, points_3D)
    """
    # Convert frames to grayscale if they aren't already
    if len(frame_left.shape) == 3:  # Color image
        gray_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
    else:  # Already grayscale
        gray_left = frame_left
        
    if len(frame_right.shape) == 3:  # Color image
        gray_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)
    else:  # Already grayscale
        gray_right = frame_right

    if debug:
        print(f"  Input image shapes: Left {gray_left.shape}, Right {gray_right.shape}")
        print(f"  Image data types: Left {gray_left.dtype}, Right {gray_right.dtype}")

    # For high-resolution images, we might want to resize for faster processing
    h, w = gray_left.shape
    if w > 2000:  # High resolution - consider downsampling for speed
        scale_factor = 0.5
        small_w, small_h = int(w * scale_factor), int(h * scale_factor)
        gray_left_small = cv2.resize(gray_left, (small_w, small_h))
        gray_right_small = cv2.resize(gray_right, (small_w, small_h))
        
        if debug:
            print(f"  Downsampled to: {gray_left_small.shape} (scale: {scale_factor})")
    else:
        gray_left_small = gray_left
        gray_right_small = gray_right
        scale_factor = 1.0

    # Improved stereo matching parameters for better results
    # For high-res images, we need more disparities and larger block size
    num_disparities = 96  # Must be divisible by 16, increased for better range
    block_size = 21       # Larger block size for high-resolution images
    
    # Create stereo matcher with better parameters
    stereo = cv2.StereoBM_create(
        numDisparities=num_disparities,
        blockSize=block_size
    )
    
    # Set additional parameters for better matching
    stereo.setUniquenessRatio(10)
    stereo.setSpeckleWindowSize(100)
    stereo.setSpeckleRange(32)
    stereo.setDisp12MaxDiff(1)
    
    if debug:
        print(f"  Stereo parameters: numDisparities={num_disparities}, blockSize={block_size}")

    # Compute the disparity map
    disparity = stereo.compute(gray_left_small, gray_right_small)
    
    if debug:
        print(f"  Raw disparity range: {np.min(disparity)} to {np.max(disparity)}")
        print(f"  Disparity shape: {disparity.shape}")

    # Scale disparity back up if we downsampled
    if scale_factor != 1.0:
        disparity = cv2.resize(disparity, (w, h))
        disparity = disparity / scale_factor  # Adjust disparity values for scale

    # Handle invalid disparities (set to 0)
    disparity[disparity <= 0] = 0
    disparity[disparity == stereo.getNumDisparities()*16] = 0
    
    # Convert to proper data type and normalize for visualization
    disparity = disparity.astype(np.float32) / 16.0  # StereoBM returns 16-bit fixed point
    
    if debug:
        valid_pixels = np.sum(disparity > 0)
        total_pixels = disparity.size
        print(f"  Valid disparity pixels: {valid_pixels}/{total_pixels} ({100*valid_pixels/total_pixels:.1f}%)")
        print(f"  Processed disparity range: {np.min(disparity[disparity > 0]):.1f} to {np.max(disparity):.1f}")

    # Create a better normalized version for visualization
    disparity_vis = disparity.copy()
    disparity_vis[disparity_vis <= 0] = np.nan  # Set invalid to NaN
    
    # Normalize only valid disparities
    valid_mask = ~np.isnan(disparity_vis)
    if np.any(valid_mask):
        min_disp = np.nanmin(disparity_vis)
        max_disp = np.nanmax(disparity_vis)
        if max_disp > min_disp:
            disparity_normalized = np.zeros_like(disparity_vis)
            disparity_normalized[valid_mask] = 255 * (disparity_vis[valid_mask] - min_disp) / (max_disp - min_disp)
        else:
            disparity_normalized = np.zeros_like(disparity_vis)
    else:
        disparity_normalized = np.zeros_like(disparity_vis)
    
    # Fill invalid areas with black
    disparity_normalized[~valid_mask] = 0

    # Create a 3D image from the disparity map
    # For real-world coordinates, you'd need actual camera calibration parameters
    baseline = 100.0  # Baseline in mm (you should measure this)
    focal_length = 0.8 * w  # Rough estimate - should come from calibration
    
    Q = np.float32([[1, 0, 0, -0.5 * w],
                     [0, -1, 0, 0.5 * h],  # Flip the y-axis
                     [0, 0, 0, -focal_length],
                     [0, 0, 1/baseline, 0]])

    # Reproject the disparity map to 3D
    points_3D = cv2.reprojectImageTo3D(disparity, Q)

    return disparity_normalized.astype(np.uint8), points_3D

def estimate_depth_sgbm(frame_left, frame_right, debug=False):
    """
    Alternative depth estimation using SGBM (Semi-Global Block Matching).
    Often produces better results than basic block matching.
    """
    # Convert frames to grayscale if they aren't already
    if len(frame_left.shape) == 3:
        gray_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
    else:
        gray_left = frame_left
        
    if len(frame_right.shape) == 3:
        gray_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)
    else:
        gray_right = frame_right

    if debug:
        print(f"  SGBM Input shapes: Left {gray_left.shape}, Right {gray_right.shape}")

    h, w = gray_left.shape
    
    # Resize for processing if too large
    if w > 2000:
        scale_factor = 0.5
        small_w, small_h = int(w * scale_factor), int(h * scale_factor)
        gray_left_small = cv2.resize(gray_left, (small_w, small_h))
        gray_right_small = cv2.resize(gray_right, (small_w, small_h))
        if debug:
            print(f"  SGBM Downsampled to: {gray_left_small.shape}")
    else:
        gray_left_small = gray_left
        gray_right_small = gray_right
        scale_factor = 1.0

    # SGBM parameters - these often work better than basic BM
    min_disparity = 0
    num_disparities = 96  # Must be divisible by 16
    block_size = 11       # Must be odd
    
    # SGBM specific parameters
    P1 = 8 * 3 * block_size**2    # Controls smoothness
    P2 = 32 * 3 * block_size**2   # Controls smoothness (larger penalty)
    
    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disparity,
        numDisparities=num_disparities,
        blockSize=block_size,
        P1=P1,
        P2=P2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    
    if debug:
        print(f"  SGBM parameters: numDisparities={num_disparities}, blockSize={block_size}")
        print(f"  P1={P1}, P2={P2}")

    # Compute disparity
    disparity = stereo.compute(gray_left_small, gray_right_small)
    
    # Scale back if downsampled
    if scale_factor != 1.0:
        disparity = cv2.resize(disparity, (w, h))
        disparity = disparity / scale_factor

    # Convert to float and handle invalid values
    disparity = disparity.astype(np.float32) / 16.0
    disparity[disparity <= 0] = 0
    disparity[disparity >= num_disparities] = 0
    
    if debug:
        valid_pixels = np.sum(disparity > 0)
        total_pixels = disparity.size
        print(f"  SGBM valid pixels: {valid_pixels}/{total_pixels} ({100*valid_pixels/total_pixels:.1f}%)")
        if valid_pixels > 0:
            print(f"  SGBM disparity range: {np.min(disparity[disparity > 0]):.1f} to {np.max(disparity):.1f}")

    # Normalize for visualization
    disparity_vis = disparity.copy()
    disparity_vis[disparity_vis <= 0] = np.nan
    
    valid_mask = ~np.isnan(disparity_vis)
    if np.any(valid_mask):
        min_disp = np.nanmin(disparity_vis)
        max_disp = np.nanmax(disparity_vis)
        if max_disp > min_disp:
            disparity_normalized = np.zeros_like(disparity_vis)
            disparity_normalized[valid_mask] = 255 * (disparity_vis[valid_mask] - min_disp) / (max_disp - min_disp)
        else:
            disparity_normalized = np.zeros_like(disparity_vis)
    else:
        disparity_normalized = np.zeros_like(disparity_vis)
    
    disparity_normalized[~valid_mask] = 0

    # Create 3D points
    baseline = 100.0
    focal_length = 0.8 * w
    Q = np.float32([[1, 0, 0, -0.5 * w],
                     [0, -1, 0, 0.5 * h],
                     [0, 0, 0, -focal_length],
                     [0, 0, 1/baseline, 0]])

    points_3D = cv2.reprojectImageTo3D(disparity, Q)

    return disparity_normalized.astype(np.uint8), points_3D 