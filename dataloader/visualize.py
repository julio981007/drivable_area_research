import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_data(image, lidar_data):
    fig = plt.figure(figsize=(15, 7))

    # 이미지 시각화
    ax1 = fig.add_subplot(121)
    ax1.imshow(image.permute(1, 2, 0).cpu().numpy())  # CHW to HWC
    ax1.set_title("Image")
    ax1.axis('off')

    # LiDAR 데이터 시각화
    ax2 = fig.add_subplot(122, projection='3d')
    
    # LiDAR 데이터를 numpy 배열로 변환
    points = lidar_data[:, :3].cpu().numpy()
    
    # 색상 설정 (높이에 따라)
    colors = plt.cm.jet(points[:, 2] / np.max(points[:, 2]))

    # 포인트 클라우드 그리기
    scatter = ax2.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=1)
    
    ax2.set_title("LiDAR Point Cloud")
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')

    # 컬러바 추가
    plt.colorbar(scatter, ax=ax2, label='Height')

    plt.tight_layout()
    plt.show()