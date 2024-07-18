import open3d as o3d
import numpy as np
from PIL import Image
import sys

def read_cam_k(file_path): # kitti
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('P0:'):
                values = np.array(line.split(':')[1].split()).astype(float)
                return np.array([[values[0], 0, values[2]],
                                 [0, values[5], values[6]],
                                 [0, 0, 1]])

if __name__ == "__main__":
    calib_file = 'kitti/calib.txt' # kitti
    cam_k = read_cam_k(calib_file) #intrinsic parameter
    print(cam_k)
    
    # depth_map
    depth_pil = Image.open("kitti/raw_depth.png")
    depth_np = np.array(depth_pil).astype(np.float32)
    depth_o3d = o3d.geometry.Image(depth_np)
    print(np.min(depth_np))
    print(np.max(depth_np))
    
    # img size
    width, height = depth_np.shape[1], depth_np.shape[0]
    
    # intrinsic parameter
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width=int(width),
        height=int(height),
        fx=cam_k[0, 0], # x focal length
        fy=cam_k[1, 1], # y focal length
        cx=cam_k[0, 2], # x principal point
        cy=cam_k[1, 2]  # y principal point
    )
    
    # pointcloud
    pcd = o3d.geometry.PointCloud.create_from_depth_image(
        depth=depth_o3d,
        intrinsic=intrinsic,
        depth_scale=1000.0,  # 실제 거리 단위로 변환하는 값
        depth_trunc=1000.0,  # 해당 값 이상의 거리는 무시
    )
    # visualize
    o3d.visualization.draw_geometries([pcd])