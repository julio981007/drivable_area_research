import open3d as o3d
import numpy as np
from PIL import Image
from gt2b import binary_convert
import os


def read_cam_k(file_path):
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('cam_K:'):
                return np.array(line.split(':')[1].split()).astype(float).reshape(3, 3)

def depth_to_pointcloud(depth_path, cam_k, binary_gt):
    # depth_map
    depth_pil = Image.open(depth_path)
    depth_np = np.array(depth_pil).astype(np.float32)
    depth_o3d = o3d.geometry.Image(depth_np)
    
    # img size
    width, height = depth_np.shape[1], depth_np.shape[0]
    
    # intrinsic parameter
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width=width,
        height=height,
        fx=cam_k[0, 0],  # x focal length
        fy=cam_k[1, 1],  # y focal length
        cx=cam_k[0, 2],  # x principal point
        cy=cam_k[1, 2]   # y principal point
    )
    
    # pointcloud
    pcd = o3d.geometry.PointCloud.create_from_depth_image(
        depth=depth_o3d,
        intrinsic=intrinsic,
        depth_scale=1000.0,  # 실제 거리 단위로 변환하는 값
        depth_trunc=1000.0,  # 해당 값 이상의 거리는 무시
    )
    
    # X 좌표 반전 (시각화 문제 임시 해결)
    points = np.asarray(pcd.points)
    points[:, 0] = -points[:, 0]  # X 좌표 반전
    pcd.points = o3d .utility.Vector3dVector(points)

    # 2차원 배열 초기화 (각 요소는 4차원 좌표를 담을 수 있는 numpy array)
    process_depth_arr = np.full((height, width), None, dtype=object)
    
    # 각 픽셀 위치에 3D 좌표와 이진 값 저장
    for i in range(height):
        for j in range(width):
            index = i * width + j
            if index < len(points):
                process_depth_arr[i, j] = np.append(points[index], binary_gt[i, j])
    
    return pcd, process_depth_arr

def display_gt_and_depth(gt_path):
    # GT 이미지 로드 및 이진화
    binary_convert(gt_path)

def save_process_depth_arr(process_depth_arr, folder_path, file_name):
    # 폴더가 없으면 생성
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    # 전체 파일 경로 생성
    file_path = os.path.join(folder_path, file_name)
    
    # 배열 저장
    np.save(file_path, process_depth_arr)