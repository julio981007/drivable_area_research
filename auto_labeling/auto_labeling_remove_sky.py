import os
from tqdm import tqdm
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

def main():
    for folder in folders:
        pl_path = os.path.join(base_path, f'{folder}/{save_folder_name}')
        
        pl_list = [file for file in os.listdir(pl_path) if file.endswith('_fillcolor.png')]

        for i in tqdm(pl_list):
            file_name = os.path.join(pl_path, i)
            gt_image = cv2.imread(file_name)
            gt_image = cv2.cvtColor(gt_image, cv2.COLOR_BGR2RGB)
            gt_image = cv2.cvtColor(gt_image, cv2.COLOR_BGR2GRAY)
            gt_image = np.array(gt_image)
            gt_image[gt_image<255] = 0
            
            print(gt_image.shape)
            plt.imshow(gt_image)
            plt.colorbar()
            plt.show()
            sys.exit()

if __name__ == "__main__":
    base_path = '/home/julio981007/HDD/orfd'
    folders = ['0', '1', '2', '3', '4', '5']
    folders = ['training', 'testing', 'validation']
    folders = ['training']
    
    save_folder_name = 'auto_labeling_raw_planarity_v2'# 'pseudo_labeling_raw_depth'
    
    main()