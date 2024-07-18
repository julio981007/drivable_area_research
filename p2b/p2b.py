import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

def read_bin_file(file_path):
    # points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4) #kitti
    points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 5) # ORFD
    return points

def create_top_view(points, resolution=0.1, x_range=(-10, 10), y_range=(-10, 10)):
    x_bins = int((x_range[1] - x_range[0]) / resolution)
    y_bins = int((y_range[1] - y_range[0]) / resolution)
    
    intensity_view = np.zeros((y_bins, x_bins), dtype=np.float32)
    count = np.zeros((y_bins, x_bins), dtype=np.int32)
    
    for point in points:
        x, y, intensity = point[0], point[1], point[3]
        if x_range[0] <= x < x_range[1] and y_range[0] <= y < y_range[1]:
            x_index = int((x - x_range[0]) / resolution)
            y_index = int((y - y_range[0]) / resolution)
            
            intensity_view[y_index, x_index] += intensity
            count[y_index, x_index] += 1
    
    # 평균 intensity 계산
    mask = count > 0
    intensity_view[mask] /= count[mask]
    
    return intensity_view

def visualize_3d(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    
    # intensity 값에 따라 색상 지정
    colors = plt.cm.viridis((points[:, 3] - np.min(points[:, 3])) / (np.max(points[:, 3]) - np.min(points[:, 3])))[:, :3]
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    o3d.visualization.draw_geometries([pcd])

def visualize_intensity_view(intensity_view):
    plt.figure(figsize=(10, 10))
    plt.imshow(intensity_view, cmap='viridis', origin='lower')
    print(intensity_view.shape)
    plt.colorbar(label='Mean Intensity')
    plt.title('LiDAR Point Cloud - Top View')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

if __name__ == "__main__":
    file_path = "pointcloudpy.bin" 
    
    # 포인트 클라우드 데이터 읽기
    points = read_bin_file(file_path)
    
    # 원본 3D 포인트 클라우드 시각화
    visualize_3d(points)
    
    # Top 뷰 생성
    intensity_view = create_top_view(points, resolution=0.1, x_range=(-30, 30), y_range=(-30, 30))
    
    # Top 뷰 시각화
    visualize_intensity_view(intensity_view)