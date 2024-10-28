import torch
import random
import numpy as np
from drivable_learner import DrivableLearner
import os
import argparse
import sys

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
parser.add_argument("--dataset_dir", type=str, required=False, default='/home/julio981007/HDD/orfd')
parser.add_argument("--dataset_folders", type=str, required=False, nargs='+', default=['training']) # ['0', '1', '2', '3', '4', '5'] / ['training']

parser.add_argument("--ckpt_dir", type=str, default="./checkpoints/orfd_AL_ESPNet(basic_p2q5)") # orfd_AL(rawdepth)_unet(basic)_nodepth

parser.add_argument("--depth", type=str2bool, default=False)
parser.add_argument("--gt_pl", type=str2bool, default=False)
parser.add_argument("--remove_sky", type=str2bool, default=True)
parser.add_argument("--labeling_folder", type=str, 
        choices=['auto_labeling', 'auto_labeling_raw_depth', 'auto_labeling_minmax_depth', 'auto_labeling_std_depth', 'auto_labeling_raw_height', 'auto_labeling_raw_planarity_v2'], 
        default="auto_labeling")

parser.add_argument("--patience", type=int, default=10)
parser.add_argument("--learning_rate", type=float, default=1e-4, help='Learning rate of for adam')
parser.add_argument("--batch_size", type=int, default=16) # 8
parser.add_argument("--num_epochs", type=int, default=50)

parser.add_argument("--img_height", type=int, default=512) # 512
parser.add_argument("--img_width", type=int, default=512) # 512

parser.add_argument("--summary_freq", type=int, default=10)
# parser.add_argument("--max_to_keep", type=int, default=5)

args = parser.parse_args()


if __name__ == '__main__':
    seed = 34
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    print(args)

    os.environ["XFORMERS_DISABLED"] = "1" # Switch to enable xFormers
    USE_CUDA = torch.cuda.is_available()
    print(USE_CUDA)
    device = torch.device('cuda:0' if USE_CUDA else 'cpu')
    if device=="cuda": torch.cuda.empty_cache()
    print('학습을 진행하는 기기:',device)
    torch.cuda.set_per_process_memory_fraction(fraction=0.8, device=device)
    
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)
    else:
        raise Exception('Already exists checkpoint directory. Please remove it or specify a different directory.')
    
    learner = DrivableLearner(args, device)
    learner.train()