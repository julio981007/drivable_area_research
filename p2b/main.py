from d2nsq import read_cam_k, binary_convert, depth_to_pointcloud, display_gt_and_depth, save_process_depth_arr
from visualize_poincloud import load_and_visualize_pointcloud

import open3d as o3d

if __name__ == "__main__":
    calib_file = 'orfd/calib.txt'  # set path
    depth_file = 'orfd/raw_depth.png'
    gt_file = 'orfd/gt_img.png'
    
    cam_k = read_cam_k(calib_file)  # read intrinsic parameter
    binary_gt = binary_convert(gt_file)
    pcd, process_depth_arr = depth_to_pointcloud(depth_file, cam_k, binary_gt)
    
    # visualize
    o3d.visualization.draw_geometries([pcd])
    display_gt_and_depth(gt_file)

    # process_depth_arr 저장
    output_folder = 'orfd'
    output_file = 'output.npy'
    save_process_depth_arr(process_depth_arr, output_folder, output_file)

    # visualize
    file_path = 'orfd/output.npy'  # 저장된 배열의 경로
    load_and_visualize_pointcloud(file_path)