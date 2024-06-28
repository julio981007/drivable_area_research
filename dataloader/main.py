from dataload import get_dataloaders
from visualize import visualize_data

def main():
    base_dir = '.'  # 현재 디렉토리
    train_loader, val_loader, test_loader = get_dataloaders(base_dir)
    
    print("Training data:")
    for images, lidar_data, gt_images in train_loader:
        print("Images shape:", images.shape)
        print("LiDAR data shape:", lidar_data.shape)
        print("Ground truth images shape:", gt_images.shape)
        
        # 첫 번째 샘플 시각화
        visualize_data(images[0], lidar_data[0])
        break

    print("\nValidation data:")
    for images, lidar_data, gt_images in val_loader:
        print("Images shape:", images.shape)
        print("LiDAR data shape:", lidar_data.shape)
        print("Ground truth images shape:", gt_images.shape)
        break

    print("\nTesting data:")
    for images, lidar_data, gt_images in test_loader:
        print("Images shape:", images.shape)
        print("LiDAR data shape:", lidar_data.shape)
        print("Ground truth images shape:", gt_images.shape)
        break

if __name__ == "__main__":
    main()