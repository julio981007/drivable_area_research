import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys

# 3.1.2 Traversable Region Detection
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

def detect_traversable_region(range_image, vertical_angles, horizontal_angles, delta_alpha, delta_beta, max_distance, min_vertical_angle):
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
    # plt.imshow(horizontal_angles)
    # plt.colorbar()
    # plt.show()
    # sys.exit()
    traversable_map = np.zeros_like(range_image)

    # Check if the vertical and horizontal angles are within the thresholds
    for r in range(vertical_angles.shape[0]):
        for c in range(horizontal_angles.shape[1]):
            if (np.abs(vertical_angles[r, c]) < delta_alpha and np.abs(horizontal_angles[r, c]) < delta_beta and range_image[r, c] < max_distance): #and vertical_angles[r, c] > min_vertical_angle):
                traversable_map[r, c] = 1  # Mark as traversable

    return traversable_map

# 3.2.1 Confidence of Traversability
def calculate_confidence(traversable_map, vertical_angles, horizontal_angles, alpha_threshold, beta_threshold, phi, k):
    """
    Calculate the confidence of each pixel's traversability using the inclination angles.
    
    Parameters:
    traversable_map (numpy array): Binary map of detected traversable regions.
    vertical_angles (numpy array): Vertical angles calculated from range image.
    horizontal_angles (numpy array): Horizontal angles calculated from range image.
    alpha_threshold (float): Threshold for vertical inclination angle.
    beta_threshold (float): Threshold for horizontal inclination angle.
    phi (float): Midpoint for confidence calculation.
    k (float): Logistic growth rate for confidence calculation.
    
    Returns:
    numpy array: Confidence map for traversable regions.
    """
    rows, cols = traversable_map.shape
    confidence_map = np.zeros((rows, cols))

    # Confidence calculation based on angular differences
    for r in range(rows-1):
        for c in range(cols-1):
            if traversable_map[r, c] == 1:
                avg_angle_diff = (np.abs(vertical_angles[r, c]) + np.abs(horizontal_angles[r, c])) / 2
                confidence = 1 - 1 / (1 + np.exp(-k * (avg_angle_diff - phi)))
                confidence_map[r, c] = confidence

    return confidence_map

if __name__ == '__main__':
    path = '/home/julio981007/HDD/orfd/training/dense_depth_anything/1623170292448.png'
    
    # Example usage
    range_image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    # range_image = np.random.uniform(0, 100, (64, 360))  # Example range image (64 vertical layers, 360 horizontal points)
    theta_v = 0.4  # Vertical resolution in degrees
    theta_h = 1.0  # Horizontal resolution in degrees

    # Step 1: Calculate angles
    vertical_angles, horizontal_angles = calculate_angles(range_image, theta_v, theta_h)

    # Step 2: Detect traversable regions
    max_distance = 70  # Example max distance to consider as traversable (adjust based on sensor range)
    min_vertical_angle = -10  # Example min vertical angle (e.g., -10 degrees to avoid sky regions)
    traversable_map = detect_traversable_region(range_image, vertical_angles, horizontal_angles, delta_alpha=0.0001, delta_beta=0.01, max_distance=max_distance, min_vertical_angle=min_vertical_angle) # 0.3

    # Step 3: Calculate confidence of traversability
    phi = 0.5  # Midpoint for slope determination
    k = 10  # Logistic growth rate for slope range
    confidence_map = calculate_confidence(traversable_map, vertical_angles, horizontal_angles, 0.3, 0.3, phi, k)

    print(np.unique(confidence_map))
    print(confidence_map.min(), confidence_map.max())
    plt.imshow(confidence_map)
    plt.colorbar()
    plt.show()
    sys.exit()
    
    print("Traversable Map:\n", traversable_map)
    print("Confidence Map:\n", confidence_map)