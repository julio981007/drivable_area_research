import io
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sympy import flatten
import torch
import torch.nn.functional as F
import os
import sys
from numpy.linalg import norm
import cv2
from glob import glob
from tqdm import tqdm
import time

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

def main():
    conf_mat = np.zeros((num_labels, num_labels), dtype=np.float64)
    for folder in folders:
        img_path = os.path.join(base_path, f'{folder}/image_data')
        gt_path = os.path.join(base_path, f'{folder}/gt_image')
        
        save_path = os.path.join(save_base_path, save_folder_name)
        
        img_list = [file for file in os.listdir(img_path) if file.endswith('.png')]

        for i in tqdm(img_list):
            # if not ((1623175227755<=int(i.split('.')[0]))&(int(i.split('.')[0])<=1623458407582))|((1623721491895<=int(i.split('.')[0]))&(int(i.split('.')[0])<=1623721600330)): # 1623175308229
            #     continue
            img_name = i
            
            pl_name = img_name.split('.')[0] + '.png' # '_fillcolor.png'
            # if not os.path.isfile(os.path.join(save_path, f'{pl_name}')):
            #     continue
            pseudo_label = cv2.imread(os.path.join(save_path, f'{pl_name}'), cv2.IMREAD_GRAYSCALE) / 255
            pseudo_label[:200, :] = 0
            # plt.imshow(pseudo_label)
            # plt.colorbar()
            # plt.show()
            # sys.exit()
            
            label_img_name = img_name.split('.')[0]+"_fillcolor.png"
            label_dir = os.path.join(gt_path, label_img_name)
            label_image = cv2.cvtColor(cv2.imread(label_dir), cv2.COLOR_BGR2RGB)
            oriHeight, oriWidth = label_image.shape[:2]
            label = np.zeros((oriHeight, oriWidth), dtype=np.uint8)
            label[label_image[:,:,2] > 200] = 1
            label[:200, :] = 0
            
            conf_mat += confusion_matrix(np.int_(label), np.int_(pseudo_label), num_labels)

    globalacc, pre, recall, F_score, iou = getScores(conf_mat)
    print ('glob acc : {0:.3f}, pre : {1:.3f}, recall : {2:.3f}, F1_score : {3:.3f}, IoU : {4:.3f}'.format(globalacc, pre, recall, F_score, iou))
    
if __name__ == "__main__":
    base_path = '/home/julio981007/HDD/orfd'
    folders = ['training', 'testing', 'validation']
    folders = ['testing']
    num_labels=2
    
    save_base_path = '/home/julio981007/HDD/inference'
    # save_base_path = os.path.join(base_path, folders[0])
    save_folder_name = 'orfd_AL_ESPNet(v2_p6q16r4)'
    
    main()