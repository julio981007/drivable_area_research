import os
from tqdm import tqdm
import sys

def main():
    for folder in folders:
        pl_path = os.path.join(base_path, f'{folder}/{save_folder_name}')
        
        pl_list = [file for file in os.listdir(pl_path) if file.endswith('.png')]

        for i in tqdm(pl_list):
            file_name = i.split('.')[0]
            
            src = os.path.join(pl_path, i)
            
            dst = file_name + '_fillcolor.png'
            dst = os.path.join(pl_path, dst)
            
            os.rename(src, dst)

if __name__ == "__main__":
    base_path = '/home/julio981007/HDD/orfd'
    folders = ['0', '1', '2', '3', '4', '5']
    folders = ['training', 'testing', 'validation']
    folders = ['training']
    
    save_folder_name = 'auto_labeling_minmax_depth'# 'pseudo_labeling_raw_depth'
    
    main()