import cv2
import os
import sys
from tqdm import tqdm
from glob import glob

def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        raise Exception('Folder already exists')

def main():
    video_names = glob(os.path.join(base_path, '*.mp4'))
    
    makedirs(save_path)
    
    for video_name in video_names:
        vidcap = cv2.VideoCapture(video_name)
        total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        success, image = vidcap.read()
        count = 0
        frame_count = 0

        with tqdm(total=total_frames, desc="Processing frames") as pbar:
            while success:
                if frame_count % (video_fps/10) == 0:  # 3프레임마다 한 번씩 이미지 저장 (30fps -> 10fps)
                    cv2.imwrite(os.path.join(save_path, "%06d.png") % count, image)
                    count += 1
                    # print('Saved frame:', count)
                
                success, image = vidcap.read()
                frame_count += 1
                pbar.update(1)  # 진행 바 업데이트
        print("Finished! Converted video to frames at 10fps")

if __name__ == '__main__':
    num_folder = 0 #  Folder index

    video_fps = 60
    
    base_path = f'/home/byeooon/HDD/gurka_test/{num_folder}'
    save_path = os.path.join(base_path, 'image_data')
    
    main()