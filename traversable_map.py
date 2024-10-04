import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys

def calculate_angles(range_image, theta_v, theta_h):
    """
    Calculate vertical and horizontal inclination angles for traversable region detection.
    
    Parameters:
    range_image (numpy array): A 2D range image from the 3D laser scanner.
    theta_v (float): Vertical resolution (angle between rows in degrees).
    theta_h (float): Horizontal resolution (angle between columns in degrees).
    
    Returns:
    tuple: Two 2D arrays representing vertical and horizontal inclination angles.
    """
    rows, cols = range_image.shape

    # Initialize vertical and horizontal angle matrices
    vertical_angles = np.zeros((rows-1, cols))
    horizontal_angles = np.zeros((rows, cols-1))

    # Calculate vertical inclination angles
    for r in range(1, rows):
        delta_x_v = np.abs(range_image[r-1, :] * np.sin(np.radians(theta_v)) - range_image[r, :] * np.sin(np.radians(theta_v)))
        delta_z_v = np.abs(range_image[r-1, :] * np.cos(np.radians(theta_v)) - range_image[r, :] * np.cos(np.radians(theta_v)))
        vertical_angles[r-1, :] = np.arctan2(delta_x_v, delta_z_v)

    # Calculate horizontal inclination angles
    for c in range(1, cols):
        delta_x_h = np.abs(range_image[:, c-1] * np.cos(np.radians(theta_h)) - range_image[:, c] * np.cos(np.radians(theta_h)))
        delta_y_h = np.abs(range_image[:, c-1] * np.sin(np.radians(theta_h)) - range_image[:, c] * np.sin(np.radians(theta_h)))
        horizontal_angles[:, c-1] = np.arctan2(delta_x_h, delta_y_h)

    return vertical_angles, horizontal_angles

def detect_traversable_region(range_image, vertical_angles, horizontal_angles, delta_alpha, delta_beta):
    """
    Detect traversable regions using vertical and horizontal inclination angles.
    
    Parameters:
    range_image (numpy array): 2D range image from the laser scanner.
    vertical_angles (numpy array): Calculated vertical inclination angles.
    horizontal_angles (numpy array): Calculated horizontal inclination angles.
    delta_alpha (float): Threshold for vertical angle difference.
    delta_beta (float): Threshold for horizontal angle difference.
    
    Returns:
    numpy array: Binary map indicating traversable regions.
    """
    traversable_map = np.zeros_like(range_image)

    # Check if the vertical and horizontal angles are within the thresholds
    for r in range(vertical_angles.shape[0]):
        for c in range(horizontal_angles.shape[1]):
            if np.abs(vertical_angles[r, c]) < delta_alpha and np.abs(horizontal_angles[r, c]) < delta_beta:
                traversable_map[r, c] = 1  # Mark as traversable

    return traversable_map

if __name__ == '__main__':
    range_image = cv2.imread('/home/julio981007/HDD/orfd/training/dense_depth/1623170291553.png', cv2.IMREAD_GRAYSCALE)
    
    # Example usage
    # range_image = np.random.uniform(0, 100, (64, 360))  # Example range image (64 vertical layers, 360 horizontal points)
    theta_v = 0.4  # Vertical resolution in degrees
    theta_h = 1.0  # Horizontal resolution in degrees

    vertical_angles, horizontal_angles = calculate_angles(range_image, theta_v, theta_h)
    # plt.imshow(horizontal_angles)
    # plt.colorbar()
    # plt.show()
    # sys.exit()
    
    traversable_map = detect_traversable_region(range_image, vertical_angles, horizontal_angles, delta_alpha=0.3, delta_beta=0.3)

    print(np.unique(traversable_map))
    plt.imshow(horizontal_angles)
    plt.colorbar()
    plt.show()
    sys.exit()