import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys
import open3d as o3d
import os
import matplotlib.colors as mcolors
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

class Depth2Voxel:
    def __init__(self, calibration_file):
        # Load calibration parameters
        self.K, self.RT, self.lidar_R, self.lidar_T = self.load_calibration(calibration_file)

    def load_calibration(self, calibration_file):
        with open(calibration_file, 'r') as f:
            lines = f.readlines()
            cam_K = np.array([float(x) for x in lines[0].split()[1:]]).reshape((3, 3))
            cam_RT = np.array([float(x) for x in lines[1].split()[1:]]).reshape((4, 4))
            lidar_R = np.array([float(x) for x in lines[2].split()[1:]]).reshape((3, 3))
            lidar_T = np.array([float(x) for x in lines[3].split()[1:]]).reshape((3, 1))
        return cam_K, cam_RT, lidar_R, lidar_T

    def load_depth_map(self, depth_map_file):
        # Load the depth map from a PNG file
        depth_map = cv2.imread(depth_map_file, cv2.IMREAD_UNCHANGED)
        
        # If the depth map is stored in 16-bit format, convert it to meters
        if depth_map.dtype == np.uint16:
            depth_map = depth_map.astype(np.float32) / 1000.0  # Assuming the depth map is in millimeters
        return depth_map

    def depth_to_point_cloud(self, depth_map):
        # Image dimensions
        h, w = depth_map.shape
        
        # Generate pixel grid coordinates
        i, j = np.indices((h, w))
        pixel_coords = np.stack((j, i, np.ones_like(i)), axis=-1)  # shape: (h, w, 3)
        
        # Reshape pixel coordinates for matrix operations
        pixel_coords = pixel_coords.reshape(-1, 3).T  # shape: (3, h * w)
        
        # Invert K matrix for transformation
        K_inv = np.linalg.inv(self.K)
        
        # Calculate normalized image coordinates
        normalized_coords = K_inv @ pixel_coords  # shape: (3, h * w)
        
        # Reshape depth map and convert it to 3D points
        depth_flat = depth_map.reshape(-1)  # shape: (h * w,)
        points_3D = normalized_coords * depth_flat  # Broadcasting depth across normalized coordinates
        
        # Apply camera extrinsics (cam_RT)
        points_3D_hom = np.vstack((points_3D, np.ones((1, points_3D.shape[1]))))  # Homogeneous coordinates
        points_3D_world = self.RT @ points_3D_hom  # shape: (4, h * w)
        
        # Convert to 3D points by removing homogeneous coordinate
        points_3D_world = points_3D_world[:3].T  # shape: (h * w, 3)

        return points_3D_world
    
    def create_point_cloud(self, points_3D):
        # Convert points to Open3D format
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_3D)
        return pcd

if __name__ == '__main__':
    img_path = '/home/julio981007/HDD/HDX/senario_5_img/' # os.path.join(base_path, f'{folder}/image_data')
    img_list = [file for file in os.listdir(img_path) if file.endswith('.png')]
    
    for file_name in img_list:
        file_name = file_name.split('.')[0]
        calibration_file = f"/home/julio981007/HDD/HDX/calib.txt"
        # depth_map_file = f"/home/julio981007/HDD/orfd/training/dense_depth_anything/{file_name}.png"
        depth_map_file = f"/home/julio981007/HDD/HDX/testing/dense_depth_anything/{file_name}.png"
        
        converter = Depth2Voxel(calibration_file)
        depth_map = converter.load_depth_map(depth_map_file)
        points_3D = converter.depth_to_point_cloud(depth_map)
        
        # o3d.visualization.draw_geometries([converter.create_point_cloud(points_3D)])
        # sys.exit()
        

        # 카메라의 내부 파라미터(내부 행렬) 설정
        fx = 266.7900  # focal length in x direction 1472.919866
        fy = 266.7725  # focal length in y direction 1452.953534
        cx = 333.4150  # principal point x-coordinate 614.779599
        cy = 185.1090  # principal point y-coordinate 353.800982
        resolution = 0.5
        camera_matrix = np.array([[fx, 0, cx],
                                [0, fy, cy],
                                [0, 0, 1]])
        
        # 이미지 크기 및 gridmap 해상도 설정
        grid_size = 50  # grid 크기, 예: 0.1m
        grid_width = 100  # gridmap 폭 (100 cells)
        grid_height = 100  # gridmap 높이 (100 cells)

        # 주행 가능 영역 정답값 (0 혹은 1로 채워진 이미지)와 depth map 불러오기
        # drivable_mask = cv2.imread(f"/home/julio981007/HDD/orfd/training/gt_image/{file_name}_fillcolor.png")  # 0 or 1
        drivable_mask = cv2.imread(f"/home/julio981007/HDD/HDX/HDX_move/velodyne_livox_1008_1023/testing/auto_labeling/{file_name}_fillcolor.png")  # /home/julio981007/HDD/HDX/HDX_move/velodyne_livox_1008_1021/testing
        oriHeight, oriWidth, _ = drivable_mask.shape
        label_image = cv2.cvtColor(drivable_mask, cv2.COLOR_BGR2RGB)
        label = np.zeros((oriHeight, oriWidth), dtype=np.uint8)
        label[label_image[:,:,2] > 200] = 1

        depth_map = cv2.imread(depth_map_file, cv2.IMREAD_GRAYSCALE)  # depth map

        # 그리드맵 초기화 (grid_size x grid_size, 해상도 단위로 설정)
        gridmap = np.zeros((int(grid_size / resolution), int(grid_size / resolution)), dtype=np.uint8)

        height, width, _ = drivable_mask.shape

        # 주행 가능 영역의 픽셀 좌표를 3D 공간의 점으로 변환
        for v in range(height):
            for u in range(width):
                if label[v, u] == 1:  # 주행 가능 영역인 경우에만 처리
                    z = depth_map[v, u]  # 깊이 값 (depth map의 값을 사용)
                    if z == 0:  # 깊이가 0이면 무시
                        continue
                    
                    # 2D 이미지 좌표를 3D 카메라 좌표계로 변환
                    x = (u - cx) * z / fx
                    y = (v - cy) * z / fy

                    # 카메라 좌표를 top-down gridmap 좌표로 변환 (x, z만 사용)
                    grid_x = int((x + grid_size / 2) / resolution)  # 그리드 중심이 0,0이 되도록 변환
                    grid_y = int((z) / resolution)

                    if 0 <= grid_x < gridmap.shape[1] and 0 <= grid_y < gridmap.shape[0]:
                        gridmap[grid_y, grid_x] = 1  # 주행 가능 영역을 gridmap에 표시
        ##########################################################################################
        # # Create a meshgrid for interpolation
        # grid_x, grid_y = np.meshgrid(np.arange(gridmap.shape[1]), np.arange(gridmap.shape[0]))
        # # Get coordinates of non-zero points
        # points = np.array(np.nonzero(gridmap)).T
        # values = gridmap[points[:, 0], points[:, 1]]
        # # Perform nearest neighbor interpolation
        # interpolated_gridmap = griddata(points, values, (grid_y, grid_x), method='nearest')
        
        # Apply Gaussian filter
        sigma = 2  # Adjust this value to control the amount of smoothing
        interpolated_gridmap = gaussian_filter(gridmap.astype(float), sigma)
        # Normalize the result
        interpolated_gridmap = (interpolated_gridmap - interpolated_gridmap.min()) / (interpolated_gridmap.max() - interpolated_gridmap.min())
        
        cmap = mcolors.ListedColormap(['white', 'green'])
        # 그리드맵 시각화 시 실제 거리 단위로 x축과 y축 설정
        plt.imshow(interpolated_gridmap, cmap=cmap, extent=[-grid_size/2, grid_size/2, grid_size, 0], vmin=0, vmax=1)
        plt.title('Top-Down Gridmap')
        plt.xlabel('X (meters)')
        plt.ylabel('Y (meters)')
        
        plt.gca().set_xticks(np.arange(-grid_size/2, grid_size/2 + resolution, resolution), minor=True)  # 세부 그리드 (resolution 단위)
        plt.gca().set_yticks(np.arange(0, grid_size + resolution, resolution), minor=True)  # 세부 그리드 (resolution 단위)
        
        # 그리드 선 표시 (세부적인 그리드 표시)
        plt.gca().grid(False, which='minor', color='gray', linestyle='-', linewidth=0.5)  # 세부 그리드 표시
        # # 주 그리드 선도 5미터 단위로 표시
        plt.gca().set_xticks(np.arange(-grid_size/2, grid_size/2 + 5, 5), minor=False)  # 주 그리드 선 (5미터 간격)
        plt.gca().set_yticks(np.arange(0, grid_size + 5, 5), minor=False)  # 주 그리드 선 (5미터 간격)
        plt.xlim(-25, 25)
        plt.ylim(0, 25)
        plt.gca().grid(True, which='major', color='gray', linestyle='-', linewidth=1.0)
        # # 주 그리드 레이블 표시
        # plt.gca().set_xticklabels(np.arange(-grid_size/2, grid_size/2 + 5, 5))  # x축 레이블을 5미터 단위로 설정
        # plt.gca().set_yticklabels(np.arange(-grid_size/2, grid_size/2 + 5, 5))  # y축 레이블을 5미터 단위로 설정
        plt.show()