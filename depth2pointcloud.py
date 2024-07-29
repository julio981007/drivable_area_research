import numpy as np
import cv2
import open3d as o3d
import matplotlib.pyplot as plt

def load_depth_map(file_path, depth_scale=1000.0):
    depth_map = cv2.imread(file_path, cv2.IMREAD_ANYDEPTH)
    return depth_map.astype(np.float32)

def read_calibration(file_path):
    calib_data = {}
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('cam_K:'):
                calib_data['K'] = np.array(line.split(':')[1].split()).astype(float).reshape(3, 3)
            elif line.startswith('cam_RT:'):
                calib_data['RT'] = np.array(line.split(':')[1].split()).astype(float).reshape(4, 4)
            elif line.startswith('lidar_R:'):
                calib_data['lidar_R'] = np.array(line.split(':')[1].split()).astype(float).reshape(3, 3)
            elif line.startswith('lidar_T:'):
                calib_data['lidar_T'] = np.array(line.split(':')[1].split()).astype(float)
    R = calib_data['RT'][:3, :3]
    T = calib_data['RT'][:3, 3]
    return calib_data['K'], R, T, calib_data['lidar_R'], calib_data['lidar_T']

def depth_to_pointcloud(depth_map, K, R, T):
    height, width = depth_map.shape
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    x = x.flatten()
    y = y.flatten()
    z = depth_map.flatten()
    valid = z > 0
    x, y, z = x[valid], y[valid], z[valid]
    points_cam = np.vstack((x, y, np.ones_like(x)))
    points_cam = np.linalg.inv(K) @ points_cam
    points_cam *= z[np.newaxis, :]
    points_world = R.T @ (points_cam - T[:, np.newaxis])
    return points_world.T

def visualize_depth_map(depth_map):
    plt.imshow(depth_map, cmap='viridis')
    plt.colorbar(label='Depth')
    plt.title('Depth Map')
    plt.show()

def visualize_pointcloud(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries([pcd])

def visualize_top_view(points):
    plt.scatter(points[:, 0], points[:, 1], s=1)
    plt.title('Top View of Point Cloud')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.axis('equal')
    plt.show()

def main():
    depth_map_path = "/home/byeooon/Desktop/p2b/orfd/1623688559852.png"
    depth_scale = 1000.0
    depth_map = load_depth_map(depth_map_path, depth_scale)
    
    # Visualize depth map
    visualize_depth_map(depth_map)
    
    calib_file = "/home/byeooon/Desktop/p2b/orfd/1623688559852.txt"
    K, R, T, _, _ = read_calibration(calib_file)
    
    point_cloud = depth_to_pointcloud(depth_map, K, R, T)
    
    # Visualize the 3D point cloud
    visualize_pointcloud(point_cloud)
    
    # Visualize top view (x-y projection)
    visualize_top_view(point_cloud[:, :2])  # Only use x and y coordinates

if __name__ == "__main__":
    main()