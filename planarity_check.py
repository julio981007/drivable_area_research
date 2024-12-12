import numpy as np
import open3d as o3d
from tqdm import tqdm
import os
import cv2
from PIL import Image
import sys
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt

from auto_labeling.utils import inv_proj, create_point_cloud, convert_to_voxel_grid, VoxelPlanarityCalculator, proj
'''
def compute_structure_tensor_planarity(points, radius):
    # 포인트 클라우드 생성
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # KD 트리 생성
    kdtree = o3d.geometry.KDTreeFlann(pcd)

    planarities = []

    for point in points:
        # 주변 점들 검색
        [k, idx, _] = kdtree.search_radius_vector_3d(point, radius)
        
        if k < 4:  # 최소 4개의 점이 필요
            planarities.append(0)
            continue

        # 주변 점들의 좌표
        neighbors = np.asarray(pcd.points)[idx[1:], :]

        # 평균 계산
        mean = np.mean(neighbors, axis=0)

        # 공분산 행렬 계산
        cov = np.cov(neighbors.T)

        # 고유값 계산
        eigenvalues, _ = np.linalg.eig(cov)
        eigenvalues = sorted(eigenvalues, reverse=True)

        # Planarity 계산
        if eigenvalues[0] == 0:
            planarity = 0
        else:
            planarity = (eigenvalues[1] - eigenvalues[2]) / eigenvalues[0]

        planarities.append(planarity)

    return np.array(planarities)
'''
def compute_structure_tensor_planarity(points, radius, k=10):
    """
    Calculate planarity for each point in the point cloud.
    
    Args:
    points (np.array): Nx3 array of point cloud coordinates
    k (int): Number of nearest neighbors to consider
    
    Returns:
    np.array: Nx1 array of planarity values
    """
    # Build KD-Tree for efficient nearest neighbor search
    tree = KDTree(points)
    
    # Find k nearest neighbors for each point
    distances, indices = tree.query(points, k=k)
    
    planarities = []
    for idx in range(len(points)):
        # Get neighboring points
        neighbors = points[indices[idx]]
        
        # Center the points
        centered_points = neighbors - np.mean(neighbors, axis=0)
        
        # Perform Singular Value Decomposition
        _, s, _ = np.linalg.svd(centered_points)
        
        # Calculate planarity
        if len(s) == 3:
            planarity = (s[1] - s[2]) / s[0]
        else:
            planarity = 0
        
        planarities.append(planarity)
    
    return np.array(planarities)

def visualize_planarity(point_cloud, planarity, planarity_threshold):
        colors = []
        planarity_arr = []

        for i in range(len(point_cloud)):
            # voxel_index = tuple(voxel_grid.get_voxel(point))
            # planarity = planarity_dict.get(voxel_index, 0)
            
            # 평탄도가 임계값 이상인 복셀만 빨간색으로 표시, 나머지는 검은색
            color = [0, 0, 0] if planarity[i] >= planarity_threshold else [1, 0, 0]
            colors.append(color)
            
            planarity_arr.append(planarity[i])

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud)
        pcd.colors = o3d.utility.Vector3dVector(np.array(colors))

        # save
        o3d.visualization.draw_geometries([pcd])

        # 저장
        pcd_np = np.asarray(pcd.points)
        colors_np = np.asarray(colors)
        planarity_np = np.asarray(planarity_arr)
        
        return pcd_np, colors_np, planarity_np

if __name__ == '__main__':
    dataset = 'orfd' # orfd, gurka, HDX
    base_path = f'/home/julio981007/HDD/{dataset}'
    # base_path = f'/home/julio981007/HDD/HDX/HDX_move/velodyne_livox_1008_1023/'
    folders = ['0', '1', '2', '3', '4', '5']
    folders = ['training', 'testing', 'validation']
    folders = ['testing']
    max_depth = 70
    planarity_threshold = 0.8
    
    for folder in tqdm(folders):
        img_path = os.path.join(base_path, f'{folder}/image_data')
        # img_path = os.path.join(base_path, f'image_data')
        depth_path = os.path.join(base_path, f'{folder}/dense_depth_anything')
        height_path = os.path.join(base_path, f'{folder}/height')
        gt_path = os.path.join(base_path, f'{folder}/gt_image')
        
        img_list = [file for file in os.listdir(img_path) if file.endswith('.png')]
        for i in tqdm(img_list):
            # if i!='1728353502251466912.png': # 1620330293343 1623175239964
            #     continue
            img_name = i
            img = Image.open(os.path.join(img_path, f'{img_name}')).convert('RGB')
            img_np = np.array(img)
            oriHeight, oriWidth, _ = img_np.shape
    
            depth = cv2.imread(os.path.join(depth_path, f'{img_name}'), cv2.IMREAD_GRAYSCALE)
            # depth = depth*(max_depth/255.)
            # print(depth.dtype)
            # sys.exit()
            depth = depth.astype(np.int16)
            depth[depth>max_depth] = -1
            # plt.imshow(depth)
            # plt.colorbar()
            # plt.show()
            # sys.exit()
            pcd = inv_proj(1280, 720).get_3dpoint_from_depthmap(depth)

            # 사용 예시
            # 임의의 3D 포인트 클라우드 생성
            np.random.seed(0)
            points = np.random.rand(1000, 3)

            # Structure tensor planarity 계산
            radius = 0.1  # 검색 반경
            planarities = compute_structure_tensor_planarity(pcd, radius)
            visualize_planarity(pcd, planarities, planarity_threshold)
            planarities[planarities>=planarity_threshold] = 1.0
            planarities[planarities<planarity_threshold] = 0.0
            
            projected_image = proj(720, 1280, max_depth).get_2dpixel_from_3dpoints(pcd, False)
            height, width, _ = projected_image.shape
            depth_drivable = planarities.reshape(height, width, 1)
            plt.imshow(depth_drivable)
            plt.colorbar()
            plt.show()

            print("Planarity 값:", planarities)
            print("평균 Planarity:", np.mean(planarities))
            print("최대 Planarity:", np.max(planarities))
            print("최소 Planarity:", np.min(planarities))
            sys.exit()