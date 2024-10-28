import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys
import open3d as o3d
import os
import json
import matplotlib.colors as mcolors
import matplotlib.patches as patches
from collections import deque

class Depth2Voxel:
    def __init__(self, calibration_file):
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
        depth_map = cv2.imread(depth_map_file, cv2.IMREAD_UNCHANGED)
        if depth_map.dtype == np.uint16:
            depth_map = depth_map.astype(np.float32) / 1000.0
        return depth_map

    def depth_to_point_cloud(self, depth_map):
        h, w = depth_map.shape
        i, j = np.indices((h, w))
        pixel_coords = np.stack((j, i, np.ones_like(i)), axis=-1)
        pixel_coords = pixel_coords.reshape(-1, 3).T
        K_inv = np.linalg.inv(self.K)
        normalized_coords = K_inv @ pixel_coords
        depth_flat = depth_map.reshape(-1)
        points_3D = normalized_coords * depth_flat
        points_3D_hom = np.vstack((points_3D, np.ones((1, points_3D.shape[1]))))
        points_3D_world = self.RT @ points_3D_hom
        points_3D_world = points_3D_world[:3].T
        return points_3D_world

    def create_point_cloud(self, points_3D):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_3D)
        return pcd

def load_bounding_boxes(json_path, score_threshold=0.2):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    filtered_bboxes = []
    for score, bbox in zip(data['scores_3d'], data['bboxes_3d']):
        if score >= score_threshold:
            filtered_bboxes.append(bbox)
    
    return filtered_bboxes
        
# 바운딩 박스 그리기
def draw_bounding_boxes(gridmap, bboxes, grid_size=50, resolution=0.5):
    for bbox in bboxes:
        x_center, y_center, z_center = bbox[:3]  # 중심 
        x_center, y_center = -1* y_center, x_center
        # y_center = y_center - 25

        # 중심 좌표가 그리드 내에 있는지 확인
        if not ((-grid_size / 2 < x_center < grid_size / 2) and (0 < y_center < grid_size/2)):
            continue  # 그리드 바깥에 있는 경우 해당 바운딩 박스 건너뛰기
        # print(x_center, y_center)

        length, width, height = bbox[3:6]  # 바운딩 박스 크기
        rotation_z = bbox[6]  # 회전 정보 (라디안)

        # # 바운딩 박스에 90도 회전 적용 (라디안으로 변환)
        # rotation_z += np.pi / 2  # 90도 회전 추가


        # 바운딩 박스 꼭짓점 계산
        corners = np.array([
            [-length / 2, -width / 2],
            [length / 2, -width / 2],
            [length / 2, width / 2],
            [-length / 2, width / 2]
        ])
        # 회전 적용 (Z축 회전, 90도 추가하여 반시계방향으로 회전)
        rotation_z += np.pi / 2  # 90도 (π/2 라디안) 추가

        # 회전 적용
        cos_angle = np.cos(rotation_z)
        sin_angle = np.sin(rotation_z)
        rotation_matrix = np.array([[cos_angle, -sin_angle], [sin_angle, cos_angle]])
        rotated_corners = np.dot(corners, rotation_matrix.T)

        # 꼭짓점 위치 변환
        # rotated_corners[:, 0] += x_center + grid_size / 2
        # rotated_corners[:, 1] += y_center + grid_size / 2

        rotated_corners[:, 0] += x_center  + grid_size / 2
        rotated_corners[:, 1] += y_center

        
        # 그리드 좌표로 변환
        grid_x = (rotated_corners[:, 0] / resolution).astype(int)
        grid_y = (rotated_corners[:, 1] / resolution).astype(int)

        # 그리드 좌표를 그리드 맵의 범위 내로 클리핑
        grid_x = np.clip(grid_x, 0, gridmap.shape[1] - 1)
        grid_y = np.clip(grid_y, 0, gridmap.shape[0] - 1)
        
        

        # print(f"grid cords{grid_x}")
        # 바운딩 박스를 그리드에 그리기
        for i in range(len(grid_x)):
            next_i = (i + 1) % len(grid_x)
            rr = np.linspace(grid_y[i], grid_y[next_i], num=100).astype(int)
            cc = np.linspace(grid_x[i], grid_x[next_i], num=100).astype(int)
            gridmap[rr, cc] = 2  # 빨간색으로 바운딩 박스 표시

            

        center_x = int((x_center + grid_size/2) / resolution)
        center_y = int((y_center) / resolution)    

        print("center")
        print(center_x, center_y)
        flood_fill(gridmap, center_y, center_x, new_color=2, boundary_color=2)

def flood_fill(gridmap, start_y, start_x, new_color, boundary_color):
    height, width = gridmap.shape
    if gridmap[start_y, start_x] == boundary_color or gridmap[start_y, start_x] == new_color:
        return  # 이미 경계 또는 색칠된 영역이면 리턴

    queue = [(start_y, start_x)]
    while queue:
        y, x = queue.pop(0)
        if 0 <= x < width and 0 <= y < height and gridmap[y, x] != boundary_color and gridmap[y, x] != new_color:
            gridmap[y, x] = new_color  # 새 색으로 채우기

            # 인접한 좌표들을 큐에 추가 (상하좌우)
            queue.append((y + 1, x))
            queue.append((y - 1, x))
            queue.append((y, x + 1))
            queue.append((y, x - 1))
            
if __name__ == '__main__':
    img_path = '/home/julio981007/HDD/HDX/selected_img/' 
    img_list = [file for file in os.listdir(img_path) if file.endswith('.png')]
    json_path = f"/home/julio981007/HDD/HDX/senario_5_preds/"
    json_list = sorted([file for file in os.listdir(json_path) if file.endswith('.json')])

# 이미지 파일 리스트와 json 파일 리스트 순서대로 처리
    for img_file, json_file in zip(img_list, json_list): #only one file
        print(f'Processing {img_file} and {json_file}')
        
        file_name = img_file.split('.')[0]
        calibration_file = f"/home/julio981007/HDD/HDX/calib.txt"
        depth_map_file = f"/home/julio981007/HDD/HDX/testing/dense_depth_anything/{file_name}.png"
        
        # JSON 파일 경로 설정
        json_full_path = os.path.join(json_path, json_file)
        
        # depth map과 JSON 파일로부터 바운딩 박스 정보 불러오기
        converter = Depth2Voxel(calibration_file)
        depth_map = converter.load_depth_map(depth_map_file)
        points_3D = converter.depth_to_point_cloud(depth_map)
        

        # o3d.visualization.draw_geometries([converter.create_point_cloud(points_3D)])
        
        fx, fy, cx, cy = 266.7900, 266.7725, 333.4150, 185.1090
        resolution, grid_size = 0.5, 50
        drivable_mask = cv2.imread(f"/home/julio981007/HDD/HDX/HDX_move/velodyne_livox_1008_1023/testing/auto_labeling/{file_name}_fillcolor.png")
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
                    
                    x = (u - cx) * z / fx
                    y = (v - cy) * z / fy

                    # 카메라 좌표를 top-down gridmap 좌표로 변환 (x, z만 사용)
                    grid_x = int((x + grid_size / 2) / resolution)  # 그리드 중심이 0,0이 되도록 변환
                    grid_y = int((z) / resolution)

                    if 0 <= grid_x < gridmap.shape[1] and 0 <= grid_y < gridmap.shape[0]:
                        gridmap[grid_y, grid_x] = 1  # 주행 가능 영역을 gridmap에 표시



        bounding_boxes = load_bounding_boxes(json_full_path)
        
        # 바운딩 박스 그리드맵에 시각화
        draw_bounding_boxes(gridmap, bounding_boxes, grid_size, resolution)
            


        ##########################################################################################
        cmap = mcolors.ListedColormap(['white', 'green', 'red'])  # 0: white, 1: green (drivable), 2: red (bounding boxes)
        plt.imshow(gridmap, cmap=cmap, extent=[-grid_size/2, grid_size/2, grid_size, 0], vmin=0, vmax=2)
        plt.title(f'Top-Down Gridmap with Bounding Boxes for {file_name}')
        plt.xlabel('X (meters)')
        plt.ylabel('Y (meters)')
        
        # plt.gca().set_xticks(np.arange(-grid_size/2, grid_size/2 + resolution, resolution), minor=True)
        # plt.gca().set_yticks(np.arange(0, grid_size + resolution, resolution), minor=True)

        # plt.gca().grid(True, which='major', color='gray', linestyle='-', linewidth=1.0)
        
        plt.gca().set_xticks(np.arange(-grid_size/2, grid_size/2 + resolution, resolution), minor=True)  # 세부 그리드 (resolution 단위)
        plt.gca().set_yticks(np.arange(0, grid_size + resolution, resolution), minor=True)  # 세부 그리드 (resolution 단위)
        
        # 그리드 선 표시 (세부적인 그리드 표시)
        plt.gca().grid(False, which='minor', color='gray', linestyle='-', linewidth=0.5)  # 세부 그리드 표시
        # # 주 그리드 선도 5미터 단위로 표시
        plt.gca().set_xticks(np.arange(-grid_size/2, grid_size/2 + 5, 5), minor=False)  # 주 그리드 선 (5미터 간격)
        plt.gca().set_yticks(np.arange(0, grid_size + 5, 5), minor=False)  # 주 그리드 선 (5미터 간격)
        plt.xlim(-25, 25)
        plt.ylim(0, 50)
        plt.gca().grid(True, which='major', color='gray', linestyle='-', linewidth=1.0)
        
        plt.xlim(-25, 25)
        plt.ylim(0, 50)
        plt.show()

# if __name__ == '__main__':
#     # 한 개의 이미지 파일과 그에 맞는 JSON 파일을 지정
#     img_file = '1728352340195674144.png'  # 시각화할 이미지 파일명
#     json_file = '1728352340195674144.json'  # 해당 이미지에 맞는 JSON 파일명
    
#     print(f'Processing {img_file} and {json_file}')
    
#     file_name = img_file.split('.')[0]
#     calibration_file = f"/home/julio981007/HDD/HDX/calib.txt"
#     depth_map_file = f"/home/julio981007/HDD/HDX/testing/dense_depth_anything/{file_name}.png"
    
#     # JSON 파일 경로 설정
#     json_full_path = os.path.join("/home/julio981007/HDD/HDX/senario_5_preds/", json_file)
    
#     # depth map과 JSON 파일로부터 바운딩 박스 정보 불러오기
#     converter = Depth2Voxel(calibration_file)
#     depth_map = converter.load_depth_map(depth_map_file)
#     points_3D = converter.depth_to_point_cloud(depth_map)
    
#     # o3d.visualization.draw_geometries([converter.create_point_cloud(points_3D)])
    
#     fx, fy, cx, cy = 266.7900, 266.7725, 333.4150, 185.1090
#     resolution, grid_size = 0.5, 50
#     drivable_mask = cv2.imread(f"/home/julio981007/HDD/HDX/HDX_move/velodyne_livox_1008_1023/testing/auto_labeling/{file_name}_fillcolor.png")
#     oriHeight, oriWidth, _ = drivable_mask.shape
#     label_image = cv2.cvtColor(drivable_mask, cv2.COLOR_BGR2RGB)
#     label = np.zeros((oriHeight, oriWidth), dtype=np.uint8)
#     label[label_image[:,:,2] > 200] = 1

#     depth_map = cv2.imread(depth_map_file, cv2.IMREAD_GRAYSCALE)  # depth map

#     # 그리드맵 초기화 (grid_size x grid_size, 해상도 단위로 설정)
#     gridmap = np.zeros((int(grid_size / resolution), int(grid_size / resolution)), dtype=np.uint8)

#     height, width, _ = drivable_mask.shape

#     # 주행 가능 영역의 픽셀 좌표를 3D 공간의 점으로 변환
#     for v in range(height):
#         for u in range(width):
#             if label[v, u] == 1:  # 주행 가능 영역인 경우에만 처리
#                 z = depth_map[v, u]  # 깊이 값 (depth map의 값을 사용)
#                 if z == 0:  # 깊이가 0이면 무시
#                     continue
                
#                 x = (u - cx) * z / fx
#                 y = (v - cy) * z / fy

#                 # 카메라 좌표를 top-down gridmap 좌표로 변환 (x, z만 사용)
#                 grid_x = int((x + grid_size / 2) / resolution)  # 그리드 중심이 0,0이 되도록 변환
#                 grid_y = int((z) / resolution)

#                 if 0 <= grid_x < gridmap.shape[1] and 0 <= grid_y < gridmap.shape[0]:
#                     gridmap[grid_y, grid_x] = 1  # 주행 가능 영역을 gridmap에 표시

#     bounding_boxes = load_bounding_boxes(json_full_path)
    
#     # 바운딩 박스 그리드맵에 시각화
#     draw_bounding_boxes(gridmap, bounding_boxes, grid_size, resolution)
        
#     ##########################################################################################
#     cmap = mcolors.ListedColormap(['white', 'green', 'red'])  # 0: white, 1: green (drivable), 2: red (bounding boxes)
#     plt.imshow(gridmap, cmap=cmap, extent=[-grid_size/2, grid_size/2, grid_size, 0], vmin=0, vmax=2)
#     plt.title(f'Top-Down Gridmap with Bounding Boxes for {file_name}')
#     plt.xlabel('X (meters)')
#     plt.ylabel('Y (meters)')
    
#     plt.gca().set_xticks(np.arange(-grid_size/2, grid_size/2 + resolution, resolution), minor=True)  # 세부 그리드 (resolution 단위)
#     plt.gca().set_yticks(np.arange(0, grid_size + resolution, resolution), minor=True)  # 세부 그리드 (resolution 단위)
    
#     plt.gca().grid(False, which='minor', color='gray', linestyle='-', linewidth=0.5)  # 세부 그리드 표시
#     plt.gca().set_xticks(np.arange(-grid_size/2, grid_size/2 + 5, 5), minor=False)  # 주 그리드 선 (5미터 간격)
#     plt.gca().set_yticks(np.arange(0, grid_size + 5, 5), minor=False)  # 주 그리드 선 (5미터 간격)
#     plt.xlim(-25, 25)
#     plt.ylim(0, 50)
#     plt.gca().grid(True, which='major', color='gray', linestyle='-', linewidth=1.0)
    
#     plt.xlim(-25, 25)
#     plt.ylim(0, 50)
#     plt.show()

