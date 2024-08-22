import io
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sympy import flatten
import torch
import torch.nn.functional as F
from torchvision.transforms import Resize, InterpolationMode
import os
import sys
from numpy.linalg import norm
import cv2
from glob import glob
from tqdm import tqdm
import time
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels
from transformers import AutoImageProcessor, AutoModel

def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)
    # else:
    #     raise Exception('Already folder exists')

def roi(model_input, box_size):
    visualize = torch.permute(model_input[0], (1,2,0)).detach().cpu().numpy()
    tmp_img = visualize.copy()
    top_width_offset = 3 # 20 # orfd : 16 / gurka : 
    left_side_width_offset = 10
    right_side_width_offset = 12

    # 이미지 자르기
    flatten_indices = []
    for i in range(box_size):
        for j in range(box_size):
            left = j * grid_size
            upper = i * grid_size
            right = left + grid_size
            lower = upper + grid_size
            if (box_size-23<=i<=box_size-22)&(box_size//2-top_width_offset<=j<=box_size//2+top_width_offset):
                flatten_indices.append(i*box_size+j)
                tmp_img[upper:lower, left:right] = np.array((255, 0, 0))
            if (box_size-10<=i<=box_size-0)&(box_size//2-left_side_width_offset-1<=j<=box_size//2-left_side_width_offset):
                flatten_indices.append(i*box_size+j)
                tmp_img[upper:lower, left:right] = np.array((255, 0, 0))
            if (box_size-10<=i<=box_size-0)&(box_size//2+right_side_width_offset<=j<=box_size//2+right_side_width_offset+1):
                flatten_indices.append(i*box_size+j)
                tmp_img[upper:lower, left:right] = np.array((255, 0, 0))
    print(box_size)
    plt.imshow(tmp_img)
    plt.savefig('tmp.png')
    sys.exit()
    return flatten_indices

def find_drivable_indices(box_size, tmp_map):
    flatten_indices = []
    for i in range(box_size):
        for j in range(box_size):
            left = j * grid_size
            upper = i * grid_size
            right = left + grid_size
            lower = upper + grid_size
            
            unique, counts = np.unique(tmp_map[upper:lower, left:right], return_counts=True)
            uniq_cnt_dict = dict(zip(unique, counts))
            if 1 in uniq_cnt_dict.keys():
                if (uniq_cnt_dict[1] / grid_size**2)>0.9:
                    flatten_indices.append(i*box_size+j)
    
    return flatten_indices

def crf(image, annot, resize_shape):
    colors, labels = np.unique(annot, return_inverse=True)
    # Example using the DenseCRF2D code
    d = dcrf.DenseCRF2D(annot.shape[1], annot.shape[0], 2)

    # get unary potentials (neg log probability)
    U = unary_from_labels(labels, 2, gt_prob=0.7, zero_unsure=False)
    d.setUnaryEnergy(U)

    # This adds the color-independent term, features are the locations only.
    d.addPairwiseGaussian(sxy=5, compat=3, kernel=dcrf.DIAG_KERNEL,
                            normalization=dcrf.NORMALIZE_SYMMETRIC)

    # This adds the color-dependent term, i.e. features are (x,y,r,g,b).
    d.addPairwiseBilateral(sxy=25, srgb=3, rgbim=np.array(image.resize(resize_shape)),
                            compat=10,
                            kernel=dcrf.DIAG_KERNEL,
                            normalization=dcrf.NORMALIZE_SYMMETRIC)

    # Run five inference steps.
    Q = d.inference(10)

    # Find out the most probable class for each pixel.
    drivable_map = np.argmax(Q, axis=0).reshape((resize_shape[1], resize_shape[0]))
    
    return drivable_map

'''
def fine_drivable(img, model_output, flatten_indices, width, height, output_size, iter):
    mean = np.mean(model_output[flatten_indices], axis=0)
    cosine_sim = np.dot(model_output, mean) / (norm(model_output)*norm(mean))
    norm_cosine = cosine_sim/np.max(cosine_sim)
    
    threshold_norm_cosine = norm_cosine.copy()
    non_drivable_indices = np.where(threshold_norm_cosine<threshold)[0]
    drivable_indices = np.where(threshold_norm_cosine>=threshold)[0]
    threshold_norm_cosine[non_drivable_indices] = 0
    threshold_norm_cosine[drivable_indices] = 1
    drivable_map = threshold_norm_cosine.reshape((output_size, output_size))
    resized = cv2.resize(drivable_map, (width, height), interpolation=cv2.INTER_NEAREST)

    crf_drivable_map = crf(img, resized, (width, height))
    
    return crf_drivable_map
'''
def fine_drivable(img, depth, model_output, flatten_indices, height, width, output_size, iter):
    mean = torch.mean(model_output[0][flatten_indices], dim=0)
    cosine_sim = F.cosine_similarity(model_output[0], mean.unsqueeze(0), dim=1)
    
    # if iter>0:
    #     depth = torch.from_numpy(depth)
    #     depth = depth.permute(1,2,0)
    #     depth = torch.flatten(depth, start_dim=0, end_dim=1)
    #     model_output_ = torch.cat((model_output[0].detach().cpu(), depth), dim=1)#.to(device)
    #     mean = torch.mean(model_output_[flatten_indices], dim=0)
    #     cosine_sim = F.cosine_similarity(model_output_, mean.unsqueeze(0), dim=1)
    
    norm_cosine = cosine_sim / torch.max(cosine_sim)
    
    threshold_norm_cosine = norm_cosine.clone()
    threshold_norm_cosine[threshold_norm_cosine < threshold] = 0
    threshold_norm_cosine[threshold_norm_cosine >= threshold] = 1
    drivable_map = threshold_norm_cosine.reshape((output_size[0], output_size[1]))
    drivable_map_np = drivable_map.detach().cpu().numpy()
    resized = cv2.resize(drivable_map_np, (width, height), interpolation=cv2.INTER_NEAREST)

    crf_drivable_map = crf(img, resized, (width, height))
    
    return crf_drivable_map

def main():
    processor = AutoImageProcessor.from_pretrained('facebook/dinov2-giant', crop_size={'height':img_size, 'width':img_size}, size={'height':img_size, 'width':img_size})
    base_path = '/media/imlab/HDD/gurka'
    folders = ['training', 'testing', 'validation']
    folders = ['0']
    # folder_idx = 0
    
    conf_mat = np.zeros((num_labels, num_labels), dtype=np.float64)
    for folder in folders:
        img_path = os.path.join(base_path, f'{folder}/image')
        # depth_path = os.path.join(base_path, f'{folder}/dense_depth')
        
        save_path = os.path.join(base_path, f'{folder}/pseudo_labeling')
        makedirs(save_path)
        
        img_list = [file for file in os.listdir(img_path) if file.endswith('.png')]
        with torch.no_grad():
            for i in tqdm(img_list):
                if i!='000149.png':
                    continue
                img_name = i
                img = Image.open(os.path.join(img_path, f'{img_name}'))
                img_np = np.array(img)
                oriHeight, oriWidth, _ = img_np.shape
                
                # depth = cv2.imread(os.path.join(depth_path, f'{img_name}'), cv2.IMREAD_GRAYSCALE)
                # depth = cv2.resize(depth, (box_size, box_size), interpolation=cv2.INTER_NEAREST)
                # depth_np = np.array(depth)
                # depth_np = (depth_np - np.mean(depth_np)) / np.std(depth_np)
                # depth_np = np.expand_dims(depth_np, axis=0)
                
                inputs = processor(images=img_np, return_tensors="pt", do_normalize=True)
                output_size = int(inputs.pixel_values[0].shape[1]/grid_size)
                
                model_input = inputs.pixel_values.to(device)
                # model_input = inputs.pixel_values
                model_output = dinov2_vitg14.get_intermediate_layers(model_input)[0]#.cpu().numpy()

                for j in range(num_iter):
                    if j==0:
                        flatten_indices = roi(model_input, box_size)
                        crf_drivable_map = fine_drivable(img, None, model_output, flatten_indices, img_size, img_size, (output_size, output_size), j)
                    else:
                        flatten_indices1 = find_drivable_indices(box_size, crf_drivable_map)
                        crf_drivable_map1 = fine_drivable(img, None, model_output, flatten_indices1, oriHeight, oriWidth, (output_size, output_size), j)
                cv2.imwrite(os.path.join(save_path, f'{img_name}'), (crf_drivable_map1*255))
    
if __name__ == "__main__":
    os.environ["XFORMERS_DISABLED"] = "1" # Switch to enable xFormers
    USE_CUDA = torch.cuda.is_available()
    print(USE_CUDA)
    device = torch.device('cuda:0' if USE_CUDA else 'cpu')
    if device=="cuda": torch.cuda.empty_cache()
    print('학습을 진행하는 기기:',device)
    dinov2_vitg14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14')
    dinov2_vitg14.eval().to(device)
    
    img_size = 644 # 644
    threshold = 0.5
    grid_size = 14
    box_size = img_size // grid_size  # num of grid per row and column
    num_labels = 2 # Drivable / Non-drivable
    num_iter = 2
    
    main()