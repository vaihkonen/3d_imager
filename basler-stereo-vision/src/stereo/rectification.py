#!/usr/bin/env python3
"""
Simple stereo rectification utility for improving disparity map quality.
This provides basic rectification without full camera calibration.
"""

import cv2
import numpy as np

def simple_rectify_stereo_pair(img_left, img_right, debug=False):
    """
    Perform simple stereo rectification using feature matching.
    This is a basic approach that doesn't require camera calibration.
    
    For best results, you should perform proper stereo calibration with
    a calibration target (checkerboard pattern).
    """
    
    if debug:
        print("  Performing simple stereo rectification...")
    
    # Convert to grayscale if needed
    if len(img_left.shape) == 3:
        gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
    else:
        gray_left = img_left
        gray_right = img_right
    
    # Detect features using ORB
    orb = cv2.ORB_create(nfeatures=2000)
    
    # Find keypoints and descriptors
    kp1, des1 = orb.detectAndCompute(gray_left, None)
    kp2, des2 = orb.detectAndCompute(gray_right, None)
    
    if des1 is None or des2 is None:
        if debug:
            print("  Warning: Could not find enough features for rectification")
        return img_left, img_right
    
    # Match features
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    
    if len(matches) < 50:
        if debug:
            print(f"  Warning: Only {len(matches)} matches found, skipping rectification")
        return img_left, img_right
    
    # Extract matched points
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    
    # Find fundamental matrix
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)
    
    if F is None:
        if debug:
            print("  Warning: Could not compute fundamental matrix")
        return img_left, img_right
    
    # Compute rectification transforms
    h, w = gray_left.shape
    _, H1, H2 = cv2.stereoRectifyUncalibrated(pts1, pts2, F, (w, h))
    
    # Apply rectification
    img_left_rect = cv2.warpPerspective(img_left, H1, (w, h))
    img_right_rect = cv2.warpPerspective(img_right, H2, (w, h))
    
    if debug:
        valid_matches = np.sum(mask)
        print(f"  Rectification completed: {valid_matches}/{len(matches)} valid matches used")
    
    return img_left_rect, img_right_rect

def align_stereo_pair_horizontally(img_left, img_right, debug=False):
    """
    Simple horizontal alignment of stereo pair using feature matching.
    This helps when cameras are not perfectly aligned.
    """
    
    if debug:
        print("  Performing horizontal alignment...")
    
    # Convert to grayscale
    if len(img_left.shape) == 3:
        gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
    else:
        gray_left = img_left
        gray_right = img_right
    
    # Detect features
    sift = cv2.SIFT_create(nfeatures=1000)
    kp1, des1 = sift.detectAndCompute(gray_left, None)
    kp2, des2 = sift.detectAndCompute(gray_right, None)
    
    if des1 is None or des2 is None:
        return img_left, img_right
    
    # Match features
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    
    # Apply ratio test
    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
    
    if len(good_matches) < 10:
        if debug:
            print(f"  Warning: Only {len(good_matches)} good matches, skipping alignment")
        return img_left, img_right
    
    # Extract matched points
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    # Calculate average vertical offset
    y_offsets = pts2[:, 0, 1] - pts1[:, 0, 1]
    median_y_offset = np.median(y_offsets)
    
    if debug:
        print(f"  Median vertical offset: {median_y_offset:.1f} pixels")
    
    # Apply vertical shift to right image if offset is significant
    if abs(median_y_offset) > 2:  # Only adjust if offset > 2 pixels
        h, w = img_right.shape[:2]
        M = np.float32([[1, 0, 0], [0, 1, -median_y_offset]])
        img_right_aligned = cv2.warpAffine(img_right, M, (w, h))
        
        if debug:
            print(f"  Applied vertical shift of {-median_y_offset:.1f} pixels to right image")
        
        return img_left, img_right_aligned
    
    return img_left, img_right
