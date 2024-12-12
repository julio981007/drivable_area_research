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

from utils import inv_proj, create_point_cloud, convert_to_voxel_grid, VoxelPlanarityCalculator, proj

def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        raise Exception('Already folder exists')

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

def fine_drivable(img, depth, lidar_drivable_map, model_output, flatten_indices, height, width, output_size, iter):
    offset = 0.2
    
    mean = torch.mean(model_output[0][flatten_indices], dim=0)
    cosine_sim = F.cosine_similarity(model_output[0], mean.unsqueeze(0), dim=1)
    
    depth_norm_cosine = torch.zeros_like(cosine_sim)
    if depth is not None:
        if iter>0:
            depth = torch.from_numpy(depth).to(device)
            depth = depth.permute(1,2,0)
            depth_flatten = torch.flatten(depth, start_dim=0, end_dim=1)
            # model_output_ = torch.cat((model_output[0], depth), dim=1)
            depth_mean = torch.mean(depth_flatten[flatten_indices], dim=0)
            offset_out = torch.logical_or((depth_mean - offset) > depth_flatten, depth_flatten > (depth_mean + offset))
            offset_in = torch.logical_not(offset_out)
            depth_cosine_sim = F.cosine_similarity(depth_flatten, depth_mean.unsqueeze(0), dim=1)
            offset_in = torch.squeeze(offset_in, dim=-1)
            offset_out = torch.squeeze(offset_out, dim=-1)
            depth_cosine_sim[offset_in] = 1.5
            depth_cosine_sim[offset_out] = -1.
            depth_norm_cosine = depth_cosine_sim
            # depth_norm_cosine = depth_cosine_sim / torch.max(depth_cosine_sim)
            
            # depth = torch.from_numpy(depth).to(device)
            # depth = depth.permute(1,2,0)
            # depth = torch.flatten(depth, start_dim=0, end_dim=1)
            # mean_ = torch.mean(depth[flatten_indices], dim=0)
            # # cosine_sim = F.cosine_similarity(depth, mean_.unsqueeze(0), dim=1)
            # depth[(mean_+0.2<depth)|(depth<mean_-0.2)] = 0
            # depth[(mean_+0.2>=depth)&(depth>=mean_-0.2)] = 1
            # depth = depth.reshape((output_size[0], output_size[1]))
            # depth_np = depth.detach().cpu().numpy()
    
    # norm_cosine = cosine_sim / torch.max(cosine_sim)
    plt.imshow(lidar_drivable_map)
    plt.colorbar()
    plt.show()
    sys.exit()
    pcd_drivable = torch.zeros_like(cosine_sim)
    if lidar_drivable_map is not None:
        pcd_drivable = cv2.resize(lidar_drivable_map, (box_size, box_size), interpolation=cv2.INTER_NEAREST)
        pcd_drivable = torch.from_numpy(pcd_drivable).to(device)
        pcd_drivable = torch.flatten(pcd_drivable, start_dim=0, end_dim=1)
        pcd_drivable = pcd_drivable / torch.max(pcd_drivable)
    
    final_norm_cosine = cosine_sim + depth_norm_cosine + 0.1*pcd_drivable
    final_norm_cosine = final_norm_cosine / torch.max(final_norm_cosine)
    
    threshold_norm_cosine = final_norm_cosine.clone()
    threshold_norm_cosine[threshold_norm_cosine < threshold] = 0
    threshold_norm_cosine[threshold_norm_cosine >= threshold] = 1
    drivable_map = threshold_norm_cosine.reshape((output_size[0], output_size[1]))
    drivable_map_np = drivable_map.detach().cpu().numpy()
    
    # if depth is not None:
    #     if iter>0:
    #         drivable_map_np = drivable_map_np.astype(np.bool_)|depth_np.astype(np.bool_)
    #         drivable_map_np = drivable_map_np.astype(np.float64)
    
    resized = cv2.resize(drivable_map_np, (width, height), interpolation=cv2.INTER_NEAREST)

    if CRF:
        resized = crf(img, resized, (width, height))
        # final_lidar_drivable_map = crf(img, lidar_drivable_map, (width, height))
        # plt.imshow(resized)
        # plt.colorbar()
        # plt.show()
        # sys.exit()
    
    return resized, final_norm_cosine.reshape((output_size[0], output_size[1])).detach().cpu().numpy()

def roi(model_input, box_size, dataset, extra_input):
    visualize = torch.permute(model_input[0], (1,2,0)).detach().cpu().numpy()
    tmp_img = visualize.copy()
    
    if dataset=='orfd':
        width_offset = 18

        flatten_indices = []
        for i in range(box_size):
            for j in range(box_size):
                left = j * grid_size
                upper = i * grid_size
                right = left + grid_size
                lower = upper + grid_size
                if (box_size-5<=i<=box_size-1)&(box_size-width_offset-1>=j>=width_offset):
                    flatten_indices.append(i*box_size+j)
                    tmp_img[upper:lower, left:right] = np.array((255, 0, 0))
    
    elif dataset=='gurka':
        top_width_offset = 2
        left_side_width_offset = 9 # Tire : 10 # Non-Tire : 13 # Snow : 
        right_side_width_offset = 10 # Tire : 12 # Non-Tire : 10 # Snow : 

        flatten_indices = []
        for i in range(box_size):
            for j in range(box_size):
                left = j * grid_size
                upper = i * grid_size
                right = left + grid_size
                lower = upper + grid_size
                if (box_size-31<=i<=box_size-30)&(box_size//2-top_width_offset<=j<=box_size//2+top_width_offset): # Tire : 24, 23 # Non-Tire : 22, 21 # Snow : 27, 26
                    flatten_indices.append(i*box_size+j)
                    tmp_img[upper:lower, left:right] = np.array((255, 0, 0))
                if (box_size-10<=i<=box_size-8)&(box_size//2-left_side_width_offset-1<=j<=box_size//2-left_side_width_offset): # Tire : 8, 4 # Non-Tire : 6, 2 # Snow : 
                    flatten_indices.append(i*box_size+j)
                    tmp_img[upper:lower, left:right] = np.array((255, 0, 0))
                if (box_size-10<=i<=box_size-8)&(box_size//2+right_side_width_offset<=j<=box_size//2+right_side_width_offset+1): # Tire : 8, 4 # Non-Tire : 6, 2 # Snow : 
                    flatten_indices.append(i*box_size+j)
                    tmp_img[upper:lower, left:right] = np.array((255, 0, 0))
                    
    elif dataset=='HDX':
        top_width_offset = 2
        left_side_width_offset = 11 # Tire : 10 # Non-Tire : 13 # Snow : 
        right_side_width_offset = 11 # Tire : 12 # Non-Tire : 10 # Snow : 

        flatten_indices = []
        for i in range(box_size):
            for j in range(box_size):
                left = j * grid_size
                upper = i * grid_size
                right = left + grid_size
                lower = upper + grid_size
                if (box_size-24<=i<=box_size-23)&(box_size//2-top_width_offset<=j<=box_size//2+top_width_offset): # Tire : 24, 23 # Non-Tire : 22, 21 # Snow : 27, 26
                    flatten_indices.append(i*box_size+j)
                    tmp_img[upper:lower, left:right] = np.array((255, 0, 0))
                if (box_size-5<=i<=box_size-4)&(box_size//2-left_side_width_offset-1<=j<=box_size//2-left_side_width_offset): # Tire : 8, 4 # Non-Tire : 6, 2 # Snow : 
                    flatten_indices.append(i*box_size+j)
                    tmp_img[upper:lower, left:right] = np.array((255, 0, 0))
                if (box_size-5<=i<=box_size-4)&(box_size//2+right_side_width_offset<=j<=box_size//2+right_side_width_offset+1): # Tire : 8, 4 # Non-Tire : 6, 2 # Snow : 
                    flatten_indices.append(i*box_size+j)
                    tmp_img[upper:lower, left:right] = np.array((255, 0, 0))
    # plt.imshow(tmp_img)
    # plt.show()
    # sys.exit()
    
    return flatten_indices

def main():
    processor = AutoImageProcessor.from_pretrained('facebook/dinov2-giant', crop_size={'height':img_size, 'width':img_size}, size={'height':img_size, 'width':img_size})
    
    conf_mat = np.zeros((num_labels, num_labels), dtype=np.float64)
    for folder in tqdm(folders):
        img_path = os.path.join(base_path, f'{folder}/image_data')
        # img_path = os.path.join(base_path, f'image_data')
        depth_path = os.path.join(base_path, f'{folder}/dense_depth_anything')
        height_path = os.path.join(base_path, f'{folder}/height')
        gt_path = os.path.join(base_path, f'{folder}/gt_image')
        
        save_path = os.path.join(base_path, f'{folder}/{save_folder_name}')
        print('\nsave path :', save_path)
        makedirs(save_path)
        
        img_list = [file for file in os.listdir(img_path) if file.endswith('.png')]
        with torch.no_grad():
            for i in tqdm(img_list):
                # if i!='1728353502251466912.png': # 1620330293343 1623175239964
                #     continue
                img_name = i
                img = Image.open(os.path.join(img_path, f'{img_name}')).convert('RGB')
                img_np = np.array(img)
                oriHeight, oriWidth, _ = img_np.shape
                
                # depth = cv2.imread(os.path.join(depth_path, f'{img_name}'), cv2.IMREAD_GRAYSCALE)
                # plt.imshow(depth)
                # plt.colorbar()
                # plt.show()
                # sys.exit()
                # depth = cv2.resize(depth, (box_size, box_size), interpolation=cv2.INTER_NEAREST)
                # depth_np = np.array(depth)
                # depth_np = depth_np*0.01 # (depth_np-depth_np.mean())/(depth_np.std()) # (depth_np - np.min(depth_np)) / (np.max(depth_np) - np.min(depth_np))
                # depth_np = np.expand_dims(depth_np, axis=0)
                
                # height = cv2.imread(os.path.join(height_path, img_name.replace('png', 'tiff')), cv2.IMREAD_UNCHANGED)
                # height = cv2.resize(height, (box_size, box_size), interpolation=cv2.INTER_NEAREST)
                # height_np = np.array(height)
                # height_np = height_np # (height_np-height_np.mean())/(height_np.std()) # (height_np - np.min(height_np)) / (np.max(height_np) - np.min(height_np))
                # height_np = np.expand_dims(height_np, axis=0)
                # depth_np = height_np
                
                lidar_drivable_map = None
                ####################################################################################
                depth = cv2.imread(os.path.join(depth_path, f'{img_name}'), cv2.IMREAD_GRAYSCALE)
                # depth = depth*(max_depth/255.)
                # print(depth.dtype)
                # sys.exit()
                depth = depth.astype(np.int16)
                depth[depth>max_depth] = -1
                # plt.imshow(depth)
                # plt.colorbar()
                # plt.show()
                # sys.exit()
                pcd = inv_proj(1280, 720).get_3dpoint_from_depthmap(depth)
                vis = create_point_cloud(pcd)
                # visualize_point_cloud(vis)
                voxel_grid = convert_to_voxel_grid(pcd, voxel_size=0.25)
                calculator = VoxelPlanarityCalculator(voxel_grid, pcd, planarity_threshold=planarity_threshold)
                planarity_dict, drivable_value = calculator.calculate_planarity()
                pcd_np, colors_np, cost_np  = calculator.visualize_planarity()
                inversed_arr = cost_np.max() + cost_np.min() - cost_np
                
                projected_image = proj(720, 1280, max_depth).get_2dpixel_from_3dpoints(pcd_np, False)
                height, width, _ = projected_image.shape
                color_mapped_image = np.zeros((height, width, 3), dtype=np.uint8)
                # color_mapped_image = colors_np.reshape(height, width, 3)
                lidar_drivable_map = color_mapped_image[:,:,0]
                
                depth_drivable = np.zeros((height, width, 1), dtype=np.uint8)
                depth_drivable = cost_np.reshape(height, width, 1)
                # plt.imshow(depth_drivable)
                # plt.colorbar()
                # plt.show()
                # sys.exit()
                ####################################################################################
                inputs = processor(images=img_np, return_tensors="pt", do_normalize=False)
                output_size = int(inputs.pixel_values[0].shape[1]/grid_size)
                
                model_input = inputs.pixel_values.to(device)
                model_output = dinov2_vitg14.get_intermediate_layers(model_input)[0]#.cpu().numpy()
                
                # min_vals = model_output.min(dim=-1, keepdim=True).values
                # max_vals = model_output.max(dim=-1, keepdim=True).values
                # scaled_tensor = (model_output - min_vals) / (max_vals - min_vals)
                # model_output = scaled_tensor

                if not extra_modality:
                    depth_np = None
                
                for j in range(num_iter):
                    if j==0:
                        flatten_indices = roi(model_input, box_size, dataset, depth_np)
                        crf_drivable_map, map = fine_drivable(img, depth_np, depth_drivable, model_output, flatten_indices, img_size, img_size, (output_size, output_size), j)
                    elif j==(num_iter-1):
                        flatten_indices = find_drivable_indices(box_size, crf_drivable_map)
                        crf_drivable_map, map = fine_drivable(img, depth_np, depth_drivable, model_output, flatten_indices, oriHeight, oriWidth, (output_size, output_size), j)
                    else:
                        flatten_indices = find_drivable_indices(box_size, crf_drivable_map)
                        crf_drivable_map, map = fine_drivable(img, depth_np, depth_drivable, model_output, flatten_indices, img_size, img_size, (output_size, output_size), j)

                resized = cv2.resize(crf_drivable_map, (oriWidth, oriHeight), interpolation=cv2.INTER_NEAREST)
                # final_lidar_drivable_map = crf(img, lidar_drivable_map, (width, height))
                # print(img_name)
                # plt.imshow(resized)
                # plt.colorbar()
                # plt.show()
                # sys.exit()

                cv2.imwrite(filename=os.path.join(save_path, img_name.split('.')[0]+'_fillcolor.png'), img=(resized*255))
                
                if dataset!='orfd':
                    continue
                
                label_img_name = img_name.split('.')[0]+"_fillcolor.png"
                label_dir = os.path.join(gt_path, label_img_name)
                label_image = cv2.cvtColor(cv2.imread(label_dir), cv2.COLOR_BGR2RGB)
                label = np.zeros((oriHeight, oriWidth), dtype=np.uint8)
                label[label_image[:,:,2] > 200] = 1
                
                conf_mat += confusion_matrix(np.int_(label), np.int_(resized), num_labels)
    
    globalacc, pre, recall, F_score, iou = getScores(conf_mat)
    print ('glob acc : {0:.3f}, pre : {1:.3f}, recall : {2:.3f}, F_score : {3:.3f}, IoU : {4:.3f}'.format(globalacc, pre, recall, F_score, iou))
    
if __name__ == "__main__":
    os.environ["XFORMERS_DISABLED"] = "1" # Switch to enable xFormers
    USE_CUDA = torch.cuda.is_available()
    print(USE_CUDA)
    device = torch.device('cuda:0' if USE_CUDA else 'cpu')
    if device=="cuda": torch.cuda.empty_cache()
    print('학습을 진행하는 기기:',device)
    dinov2_vitg14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14')
    dinov2_vitg14.eval().to(device)
    torch.cuda.set_per_process_memory_fraction(fraction=0.5, device=device)
    
    img_size = 644
    max_depth = 50
    threshold = 0.55 # orfd : 0.55 / gurka : 0.6
    planarity_threshold = 1.0
    offset = 0.3
    grid_size = 14
    box_size = img_size // grid_size  # num of grid per row and column
    num_labels = 2 # Drivable / Non-drivable
    num_iter = 2
    
    dataset = 'orfd' # orfd, gurka, HDX
    base_path = f'/home/julio981007/HDD/{dataset}'
    # base_path = f'/home/julio981007/HDD/HDX/HDX_move/velodyne_livox_1008_1023/'
    folders = ['0', '1', '2', '3', '4', '5']
    folders = ['training', 'testing', 'validation']
    folders = ['testing']
    
    save_folder_name = 'auto_labeling_raw_planarity_v4_01' # auto_labeling / auto_labeling_raw_depth
    extra_modality = False
    CRF = True
    
    main()