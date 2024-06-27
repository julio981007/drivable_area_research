import io
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
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

os.environ["XFORMERS_DISABLED"] = "1" # Switch to enable xFormers
USE_CUDA = torch.cuda.is_available()
print(USE_CUDA)
device = torch.device('cuda:0' if USE_CUDA else 'cpu')
if device=="cuda": torch.cuda.empty_cache()
print('학습을 진행하는 기기:',device)
dinov2_vitg14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14')
dinov2_vitg14.eval().to(device)

def roi(model_input, box_size):
    visualize = torch.permute(model_input[0], (1,2,0)).detach().numpy()
    tmp_img = visualize.copy()
    width_offset = 16

    # 이미지 자르기
    flatten_indices = []
    for i in range(box_size):
        for j in range(box_size):
            left = j * grid_size
            upper = i * grid_size
            right = left + grid_size
            lower = upper + grid_size
            if (box_size-9<=i<=box_size-1)&(box_size-width_offset-1>=j>=width_offset):
                flatten_indices.append(i*box_size+j)
                tmp_img[upper:lower, left:right] = np.array((255, 0, 0))
                
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

def confusion_matrix(x, y, n, ignore_label=None, mask=None):
    if mask is None:
        mask = np.ones_like(x) == 1
    k = (x >= 0) & (y < n) & (x != ignore_label) & (mask.astype(np.bool_))
    return np.bincount(n * x[k].astype(int) + y[k], minlength=n**2).reshape(n, n)

def getScores(conf_matrix):
    if conf_matrix.sum() == 0:
        return 0, 0, 0, 0, 0
    with np.errstate(divide='ignore',invalid='ignore'):
        globalacc = np.diag(conf_matrix).sum() / np.float32(conf_matrix.sum())
        classpre = np.diag(conf_matrix) / conf_matrix.sum(0).astype(np.float32)
        classrecall = np.diag(conf_matrix) / conf_matrix.sum(1).astype(np.float32)
        IU = np.diag(conf_matrix) / (conf_matrix.sum(1) + conf_matrix.sum(0) - np.diag(conf_matrix)).astype(np.float32)
        pre = classpre[1]
        recall = classrecall[1]
        iou = IU[1]
        F_score = 2*(recall*pre)/(recall+pre)
    return globalacc, pre, recall, F_score, iou

def fine_drivable(img, model_output, flatten_indices, width, height, output_size):
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
    

img_size = 280 # 644
threshold = 0.6
grid_size = 14
num_labels = 2
num_iter = 2
def main():
    processor = AutoImageProcessor.from_pretrained('facebook/dinov2-giant', crop_size={'height':img_size, 'width':img_size}, size={'height':img_size, 'width':img_size})
    base_path = '/media/imlab/HDD/ORFD'
    folders = ['training', 'testing', 'validation']
    folder_idx = 0
    img_path = os.path.join(base_path, f'{folders[folder_idx]}/image_data/')
    gt_path = os.path.join(base_path, f'{folders[folder_idx]}/gt_image/')
    
    img_list = [file for file in os.listdir(img_path) if file.endswith('.png')]
    
    conf_mat = np.zeros((num_labels, num_labels), dtype=np.float64)
    for i in tqdm(img_list):
        img_name = i
        img = Image.open(f'{img_path}{img_name}')
        img_np = np.array(img)
        oriHeight, oriWidth, _ = img_np.shape
        
        inputs = processor(images=img_np, return_tensors="pt")
        output_size = int(inputs.pixel_values[0].shape[1]/grid_size)
        box_size = img_size // grid_size  # num of grid per row and column
        
        model_input = torch.tensor(inputs.pixel_values).to(device)
        # model_input = inputs.pixel_values
        model_output = dinov2_vitg14.get_intermediate_layers(model_input)[0][0]#.detach().numpy()

        for j in tqdm(range(num_iter)):
            if j==0:
                flatten_indices = roi(model_input, box_size)
                crf_drivable_map = fine_drivable(img, model_output, flatten_indices, img_size, img_size, output_size)
            else:
                flatten_indices1 = find_drivable_indices(box_size, crf_drivable_map)
                crf_drivable_map1 = fine_drivable(img, model_output, flatten_indices1, oriWidth, oriHeight, output_size)
        
        useDir = os.path.join(base_path, folders[folder_idx])
        label_img_name = img_name.split('.')[0]+"_fillcolor.png"
        label_dir = os.path.join(useDir, 'gt_image', label_img_name)
        label_image = cv2.cvtColor(cv2.imread(label_dir), cv2.COLOR_BGR2RGB)
        # resized_label_image = cv2.resize(label_image, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
        label = np.zeros((oriHeight, oriWidth), dtype=np.uint8)
        label[label_image[:,:,2] > 200] = 1
        
        conf_mat += confusion_matrix(np.int_(label), np.int_(crf_drivable_map1), num_labels)

    globalacc, pre, recall, F_score, iou = getScores(conf_mat)
    print ('glob acc : {0:.3f}, pre : {1:.3f}, recall : {2:.3f}, F_score : {3:.3f}, IoU : {4:.3f}'.format(globalacc, pre, recall, F_score, iou))
    
if __name__ == "__main__":
    main()