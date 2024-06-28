import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

def make_transform(smaller_edge_size: int) -> transforms.Compose:
    IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
    interpolation_mode = transforms.InterpolationMode.BICUBIC
    return transforms.Compose([
        transforms.Resize(size=smaller_edge_size, interpolation=interpolation_mode, antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    ])

def bin_to_numpy(bin_path):
    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 5)[:,:4]  #jk hesai40 data
    points[:, 3] = 1.0  # homogeneous
    return points

class DrivableAreaDataset(Dataset):
    def __init__(self, base_dir, split, transform=None):
        self.image_dir = os.path.join(base_dir, split, 'image_data')
        self.lidar_dir = os.path.join(base_dir, split, 'lidar_data')
        self.gt_dir = os.path.join(base_dir, split, 'gt_image')
        self.transform = transform
        self.samples = [f for f in os.listdir(self.image_dir) if f.endswith('.png')]
        
        print(f"Dataset initialized for {split}")
        print(f"Image directory: {self.image_dir}")
        print(f"LiDAR directory: {self.lidar_dir}")
        print(f"Ground Truth directory: {self.gt_dir}")
        print(f"Number of samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name = self.samples[idx]
        img_path = os.path.join(self.image_dir, img_name)
        lidar_path = os.path.join(self.lidar_dir, img_name.replace('.png', '.bin'))
        gt_path = os.path.join(self.gt_dir, img_name.replace('.png', '_fillcolor.png'))

        # 이미지 로드
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        # LiDAR 데이터 로드
        lidar_data = bin_to_numpy(lidar_path)
        print('liader : ', lidar_data.shape)
        lidar_tensor = torch.from_numpy(lidar_data).float()

        # Ground Truth 이미지 로드
        gt_image = Image.open(gt_path).convert('L')  # 그레이스케일로 변환
        gt_tensor = transforms.ToTensor()(gt_image)

        if idx == 0:  # 첫 번째 샘플에 대해서만 shape 출력
            print(f"LiDAR data shape: {lidar_tensor.shape}")

        return image, lidar_tensor, gt_tensor

def get_dataloaders(base_dir, batch_size=1, num_workers=4):
    transform = make_transform(smaller_edge_size=224)
    train_dataset = DrivableAreaDataset(base_dir, 'training', transform=transform)
    val_dataset = DrivableAreaDataset(base_dir, 'validation', transform=transform)
    test_dataset = DrivableAreaDataset(base_dir, 'testing', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader