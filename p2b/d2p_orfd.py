import open3d as o3d
import numpy as np
from PIL import Image
from gt2b import binary_convert, display_image

def read_cam_k(file_path):
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('cam_K:'):
                return np.array(line.split(':')[1].split()).astype(float).reshape(3, 3)

def depth_to_pointcloud(depth_path, cam_k):
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
    
    # X 좌표 반전
    points = np.asarray(pcd.points)
    points[:, 0] = -points[:, 0]  # X 좌표 반전
    pcd.points = o3d.utility.Vector3dVector(points)
    
    return pcd

if __name__ == "__main__":
    calib_file = 'orfd/calib.txt'  # ORFD
    depth_file = 'orfd/raw_depth.png'
    
    cam_k = read_cam_k(calib_file)  # read intrinsic parameter
    pcd = depth_to_pointcloud(depth_file, cam_k)
    
    # visualize
    o3d.visualization.draw_geometries([pcd])