mport cv2
import numpy as np

def identify_flag(image_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Resize the image to a fixed size for consistency (adjust if necessary)
    height, width = 300, 600  # Flag aspect ratio
    image = cv2.resize(image, (width, height))

    # Split the image into top and bottom halves
    top_half = image[:height // 2, :]
    bottom_half = image[height // 2:, :]

    # Compute average color for each half (BGR format)
    top_avg_color = np.mean(top_half, axis=(0, 1))
    bottom_avg_color = np.mean(bottom_half, axis=(0, 1))

    print(f"Top half average color: {top_avg_color}")
    print(f"Bottom half average color: {bottom_avg_color}")

    # Check if top is red and bottom is white (Indonesia flag)
    if is_red(top_avg_color) and is_white(bottom_avg_color):
        return "Indonesia Flag"

    # Check if top is white and bottom is red (Poland flag)
    elif is_white(top_avg_color) and is_red(bottom_avg_color):
        return "Poland Flag"

    # If neither, return unknown
    return "Unknown Flag"

def is_red(color):
    b, g, r = color
    # Red color check (R > 130 and G, B are low)
    return r > 130 and g < 100 and b < 100

def is_white(color):
    b, g, r = color
    # White color check (R, G, B all high)
    return r > 180 and g > 180 and b > 180

# Input image path (change this to the path of your flag image)
input_image_path = r"C:\Users\LENOVO\Downloads\flag.jpg"

try:
    result = identify_flag(input_image_path)
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {e}")
