import os
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
import cv2
import numpy as np
import torch

from Depth_Anything_V2.metric_depth.depth_anything_v2.dpt import DepthAnythingV2

def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        raise Exception('Already folder exists')

def main():
    for folder in tqdm(folders):
        save_path = os.path.join(base_path, f'{folder}/{save_folder_name}')
        makedirs(save_path)
        
        img_path = os.path.join(base_path, f'{folder}/image_data')
        img_list = [file for file in os.listdir(img_path) if file.endswith('.png')]
        
        for i in tqdm(img_list):
            img_name = i
            image = cv2.imread(os.path.join(img_path, f'{img_name}'))
            
            depth = model.infer_image(image) # HxW raw depth map in numpy
            
            cv2.imwrite(os.path.join(save_path, f'{img_name}'), depth)
            
            # np.save(os.path.join(save_path, i.split('.')[0]), output)
            
if __name__=='__main__':
    os.environ["XFORMERS_DISABLED"] = "1" # Switch to enable xFormers
    USE_CUDA = torch.cuda.is_available()
    print(USE_CUDA)
    device = torch.device('cuda:0' if USE_CUDA else 'cpu')
    if device=="cuda": torch.cuda.empty_cache()
    print('학습을 진행하는 기기:',device)

    model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
    
    base_path = '/home/julio981007/HDD/HDX'
    folders = ['training', 'testing', 'validation'] # ['1', '2', '3', '4', '5']
    folders = ['testing']
    save_folder_name = 'dense_depth_anything'
    
    encoder = 'vitl' # or 'vits', 'vitb', 'vitg'
    dataset = 'vkitti' # 'hypersim' for indoor model, 'vkitti' for outdoor model
    max_depth = 80 # 20 for indoor model, 80 for outdoor model
    
    model = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': max_depth})
    model.load_state_dict(torch.load(f'./Depth_Anything_V2/metric_depth/checkpoints/depth_anything_v2_metric_{dataset}_{encoder}.pth', map_location=device))
    model = model.to(device).eval()
    
    main()