import os
import sys
from natsort import natsorted
import numpy as np
import open3d as o3d

def show_open3d_pcd(pcd_list, show_origin=True, origin_size=3, show_grid=True):
    geometries = []
    v3d = o3d.utility.Vector3dVector
    for i, pcd in enumerate(pcd_list):
        if isinstance(pcd, np.ndarray):
            cloud = o3d.geometry.PointCloud()
            cloud.points = v3d(pcd)
            if i == 0:  # 첫 번째 포인트 클라우드(주행궤적)를 빨간색으로 설정
                cloud.paint_uniform_color([1, 0, 0])
            geometries.append(cloud)
        elif isinstance(pcd, o3d.geometry.PointCloud):
            if i == 0:  # 첫 번째 포인트 클라우드(주행궤적)를 빨간색으로 설정
                pcd.paint_uniform_color([1, 0, 0])
            geometries.append(pcd)
    
    if show_origin:
        coord = o3d.geometry.TriangleMesh().create_coordinate_frame(size=origin_size, origin=np.array([0.0, 0.0, 0.0]))
        geometries.append(coord)
    
    o3d.visualization.draw_geometries(geometries)
    
base_path = os.path.dirname(os.path.abspath(__file__))

# 파일 경로 지정
file_path = os.path.join(base_path, 'lidarglob/data/')
files = natsorted(os.listdir(file_path))

matrix = []

for file in files:
    # 파일 열기
    with open(os.path.join(file_path, file), 'r') as f:
        lines = f.readlines()
    # 각 줄을 문자열에서 행렬 데이터 부분만 추출하여 리스트에 저장
    matrix_data = []
    num=0
    for line in lines:
        if num==0:
            start_index = line.find('[') + 2
            num=1
        # "["와 "]" 사이의 문자열을 추출
        else:
            start_index = line.find('[') + 1
        end_index = line.find(']')
        data_str = line[start_index:end_index]
        
        # 공백을 기준으로 문자열을 분할하고 float 형태로 변환하여 리스트에 저장
        row = [float(num) for num in data_str.split()]
        matrix_data.append(row)

    # 넘파이 배열로 변환
    matrix.append(matrix_data)
arrays = np.array(matrix[5:])

abs_coord = arrays[0, :3, 3]
relative_coords = arrays[:, :3, 3]
result = relative_coords - np.repeat(abs_coord.reshape(1,3), 23, 0)
result[:,1] = -(result[:,1])

base_path = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(base_path, 'pointcloud64/data/')
files = natsorted(os.listdir(file_path))

cnt=0
for file in files:
    pcd_path = os.path.join(file_path, file)
    pcd = np.loadtxt(pcd_path, delimiter=',')
    
    if cnt<5:
        cnt+=1
        continue
    
    show_open3d_pcd([result, pcd])
    sys.exit()