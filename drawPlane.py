import os
import sys
from natsort import natsorted
import numpy as np
import open3d as o3d

def expand_trajectory(trajectory, width=1.0):
    # 궤적을 입력받아 폭을 가진 메쉬로 확장하는 함수
    expanded_points = []
    for i in range(len(trajectory) - 1):
        # 궤적의 인접한 두 점을 사용하여 폭을 계산하고 확장된 점들을 생성
        x1, y1, z1 = trajectory[i]
        x2, y2, z2 = trajectory[i + 1]
        dx = x2 - x1
        dy = y2 - y1
        length = np.sqrt(dx**2 + dy**2)
        nx = -dy / length
        ny = dx / length
        left_x1 = x1 + nx * width
        left_y1 = y1 + ny * width
        right_x1 = x1 - nx * width
        right_y1 = y1 - ny * width
        left_x2 = x2 + nx * width
        left_y2 = y2 + ny * width
        right_x2 = x2 - nx * width
        right_y2 = y2 - ny * width
        expanded_points.append([left_x1, left_y1, z1])
        expanded_points.append([right_x1, right_y1, z1])
        expanded_points.append([left_x2, left_y2, z2])
        expanded_points.append([right_x2, right_y2, z2])
    
    expanded_points = np.array(expanded_points)
    
    # 확장된 점들을 이용하여 삼각형 메쉬 생성
    triangles = []
    for i in range(0, len(expanded_points) - 2, 2):
        triangles.append([i, i + 1, i + 2])
        triangles.append([i + 1, i + 3, i + 2])
    
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(expanded_points)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.paint_uniform_color([0, 1, 0])  # 녹색으로 색상 지정
    
    return mesh

def show_open3d_pcd(pcd_list, show_origin=True, origin_size=3, show_grid=True):
    # Open3D로 점구름 및 메쉬를 시각화하는 함수
    geometries = []
    v3d = o3d.utility.Vector3dVector
    
    for i, pcd in enumerate(pcd_list):
        if isinstance(pcd, o3d.geometry.TriangleMesh):
            geometries.append(pcd)
        elif isinstance(pcd, np.ndarray):
            cloud = o3d.geometry.PointCloud()
            cloud.points = v3d(pcd)
            geometries.append(cloud)
        elif isinstance(pcd, o3d.geometry.PointCloud):
            geometries.append(pcd)
    
    if show_origin:
        coord = o3d.geometry.TriangleMesh().create_coordinate_frame(size=origin_size, origin=np.array([0.0, 0.0, 0.0]))
        geometries.append(coord)
    
    o3d.visualization.draw_geometries(geometries)

base_path = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(base_path, 'lidarglob/data/')
files = natsorted(os.listdir(file_path))

matrix = []
for file in files:
    # lidarglob/data 폴더의 파일들을 읽어 행렬 데이터로 변환하여 matrix 리스트에 저장
    with open(os.path.join(file_path, file), 'r') as f:
        lines = f.readlines()
        matrix_data = []
        num = 0
        for line in lines:
            if num == 0:
                start_index = line.find('[') + 2
                num = 1
            else:
                start_index = line.find('[') + 1
            end_index = line.find(']')
            data_str = line[start_index:end_index]
            row = [float(num) for num in data_str.split()]
            matrix_data.append(row)
        matrix.append(matrix_data)

arrays = np.array(matrix[5:])
abs_coord = arrays[0, :3, 3]
relative_coords = arrays[:, :3, 3]
result = relative_coords - np.repeat(abs_coord.reshape(1, 3), 23, 0)
result[:, 1] = -(result[:, 1])

base_path = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(base_path, 'pointcloud64/data/')
files = natsorted(os.listdir(file_path))

cnt = 0
for file in files:
    # pointcloud64/data 폴더의 파일들을 읽어 점구름 데이터로 변환하여 시각화
    pcd_path = os.path.join(file_path, file)
    pcd = np.loadtxt(pcd_path, delimiter=',')
    
    if cnt < 5:
        cnt += 1
        continue
    
    expanded_trajectory = expand_trajectory(result, width=1.0)
    show_open3d_pcd([expanded_trajectory, pcd])
    
    sys.exit()