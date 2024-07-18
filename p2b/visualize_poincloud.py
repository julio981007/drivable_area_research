import numpy as np
import open3d as o3d

def load_and_visualize_pointcloud(file_path):
    # NumPy 배열 로드
    loaded_arr = np.load(file_path, allow_pickle=True)
    
    # 유효한 포인트만 추출 (None이 아닌 값)
    valid_points = [point for row in loaded_arr for point in row if point is not None]
    
    if not valid_points:
        print("No valid points found in the loaded array")
        return
    
    # NumPy 배열로 변환
    points_array = np.array(valid_points)
    
    # x, y, z 좌표 추출
    xyz = points_array[:, :3]
    
    # binary info 추출 (0 또는 255로 가정)
    binary_info = points_array[:, 3]
    
    # Open3D PointCloud 객체 생성
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    
    # binary info에 따라 색상 지정
    colors = np.zeros((len(xyz), 3))
    colors[binary_info == 255] = [1, 0, 0]  # 255는 빨간색
    colors[binary_info == 0] = [0, 0, 1]    # 0은 파란색
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # 포인트 클라우드 시각화
    o3d.visualization.draw_geometries([pcd])

if __name__ == "__main__":
    file_path = 'orfd/output.npy'  # 저장된 배열의 경로
    load_and_visualize_pointcloud(file_path)