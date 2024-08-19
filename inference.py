import torch
import torch.nn.functional as F
import os
import argparse
import sys
import cv2
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from nets import DrivableNet, UNet_small

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir", type=str, required=False, default='/home/julio981007/HDD/orfd/testing')
parser.add_argument("--ckpt_dir", type=str, default="./checkpoints/orfd_AL(rawheight)_unet(small)_nodepth/model_20240815_110837_27")
parser.add_argument("--save_folder", type=str, default="orfd_AL(rawheight)_unet(small)_nodepth")
parser.add_argument("--save_dir", type=str, default="/home/julio981007/HDD/inference")

parser.add_argument("--img_height", type=int, default=512) # 644
parser.add_argument("--img_width", type=int, default=512) # 644
parser.add_argument("--depth", type=str2bool, default=False)

args = parser.parse_args()

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

def make_rgb_transform(image) -> torch.Tensor:
    smaller_edge_size = (args.img_height, args.img_width)

    interpolation_mode = transforms.InterpolationMode.BICUBIC

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(size=smaller_edge_size, interpolation=interpolation_mode, antialias=True),
        transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    ])

    return transform(image)

def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        raise Exception('Already folder exists')

def main():
    num_patch = args.img_height // 14
    
    checkpoint = torch.load(args.ckpt_dir)
    # model = DrivableNet(args.depth, num_patch, device=device)
    model = UNet_small().to(device=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()
    ######################################################################
    img_folder = os.path.join(args.dataset_dir, 'image_data')
    
    img_list = os.listdir(img_folder)
    
    save_path = os.path.join(args.save_dir, args.save_folder)
    makedirs(save_path)
    print(save_path)
    
    for img in tqdm(img_list):
        img_path = os.path.join(img_folder, img)
        
        raw_image = cv2.imread(img_path)
        image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
        image = transforms.ToPILImage()(image)
        
        image = make_rgb_transform(image)
        if not isinstance(image, torch.Tensor):
            image = transforms.ToTensor()(image)
        image = image.unsqueeze(0)
        
        depth = None
        # if args.depth:
        #     depth = depth_transform(depth)
        # if not isinstance(depth, torch.Tensor):
        #     depth = transforms.ToTensor()(depth)
        
        image = image.to(device)
        out = model(image)
        out = F.interpolate(out, size=raw_image.shape[:2], mode='nearest')# , align_corners=True)
        out = (out >= torch.FloatTensor([0.5]).to(device))
        out_numpy = out.permute(0, 2, 3, 1).detach().cpu().numpy()[0]
        
        cv2.imwrite(filename=os.path.join(save_path, f'{img}'), img=(out_numpy*255))

if __name__ == '__main__':
    os.environ["XFORMERS_DISABLED"] = "1" # Switch to enable xFormers
    USE_CUDA = torch.cuda.is_available()
    print(USE_CUDA)
    device = torch.device('cuda:0' if USE_CUDA else 'cpu')
    if device=="cuda": torch.cuda.empty_cache()
    print('학습을 진행하는 기기:',device)
    torch.cuda.set_per_process_memory_fraction(fraction=0.5, device=device)
    
    main()