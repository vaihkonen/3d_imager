import cv2
import numpy as np

def preprocess_image(image):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to the image
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    
    # Normalize the image
    normalized_image = cv2.normalize(blurred_image, None, 0, 255, cv2.NORM_MINMAX)
    
    return normalized_image

def resize_image(image, width, height):
    # Resize the image to the specified width and height
    resized_image = cv2.resize(image, (width, height))
    
    return resized_image

def stack_images(image1, image2):
    # Stack two images horizontally
    stacked_image = cv2.hconcat([image1, image2])
    
    return stacked_image