import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
import cv2
import sys

# IOU, Precision, Recall 계산 함수
def calculate_metrics(rgb, pred_mask, gt_mask):
    pred_visual = (pred_mask * 255).astype(np.uint8)
    pred_visual = np.stack([pred_visual]*3, axis=-1)  # Make it 3-channel
    pred_visual[:,:,:1]=0
    
    gt_visual = (gt_mask * 255).astype(np.uint8)
    gt_visual = np.stack([gt_visual]*3, axis=-1)  # Make it 3-channel
    gt_visual[:,:,1:]=0
    
    intersection = np.logical_and(pred_mask, gt_mask)
    intersection_visual = (intersection * 255).astype(np.uint8)
    intersection_visual = np.stack([intersection_visual]*3, axis=-1)  # Make it 3-channel
    # intersection_visual[:,:,:1]=0
    union = np.logical_or(pred_mask, gt_mask)
    
    iou = np.sum(intersection) / np.sum(union)
    
    tp = np.sum(np.logical_and(pred_mask, gt_mask))  # True Positive
    fp = np.sum(np.logical_and(pred_mask, np.logical_not(gt_mask)))  # False Positive
    fn = np.sum(np.logical_and(np.logical_not(pred_mask), gt_mask))  # False Negative
    
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0
    
    tp_visual = np.logical_and(pred_mask, gt_mask)
    tp_visual = (tp_visual * 255).astype(np.uint8)
    tp_visual = np.stack([tp_visual]*3, axis=-1)  # Make it 3-channel
    tp_visual[:,:,:1]=0
    
    
    
    alpha = 1.0
    dst = cv2.addWeighted(pred_visual, alpha, tp_visual, (1-alpha), 0)
    alpha = 0.3
    dst1 = cv2.addWeighted(rgb, alpha, dst, (1-alpha), 0)
    plt.imshow(dst1)
    plt.show()
    sys.exit()
    return iou, precision, recall

# 시각화 함수
def visualize_metrics(rgb_img, pred_mask, gt_mask):
    file_name = '1623721521591'
    
    image = cv2.imread(f'/home/julio981007/HDD/orfd/testing/image_data/{file_name}.png')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.array(image)
    
    pred_mask = cv2.imread(f'/home/julio981007/HDD/orfd/testing/auto_labeling/{file_name}_fillcolor.png')
    oriHeight, oriWidth, _ = pred_mask.shape
    pred_mask = cv2.cvtColor(pred_mask, cv2.COLOR_BGR2RGB)
    pred = np.zeros((oriHeight, oriWidth), dtype=np.uint8)
    pred[pred_mask[:,:,2] > 200] = 1
    
    gt_mask = cv2.imread(f'/home/julio981007/HDD/orfd/testing/gt_image/{file_name}_fillcolor.png')
    oriHeight, oriWidth, _ = gt_mask.shape
    gt_mask = cv2.cvtColor(gt_mask, cv2.COLOR_BGR2RGB)
    gt = np.zeros((oriHeight, oriWidth), dtype=np.uint8)
    gt[gt_mask[:,:,2] > 200] = 1
    
    iou, precision, recall = calculate_metrics(image, pred, gt)
    
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # RGB 이미지 표시
    axs[0].imshow(rgb_img)
    axs[0].set_title('RGB Image')
    axs[0].axis('off')

    # 예측 마스크 투명하게 겹치기
    axs[1].imshow(rgb_img)
    axs[1].imshow(pred_mask, cmap='Reds', alpha=0.5)  # Red color for predicted mask
    axs[1].set_title(f'Predicted Mask\nIoU: {iou:.3f}, Precision: {precision:.3f}')
    axs[1].axis('off')

    # Ground Truth 마스크 투명하게 겹치기
    axs[2].imshow(rgb_img)
    axs[2].imshow(gt_mask, cmap='Blues', alpha=0.5)  # Blue color for ground truth mask
    axs[2].set_title(f'Ground Truth Mask\nRecall: {recall:.3f}')
    axs[2].axis('off')

    plt.tight_layout()
    plt.show()

# 예시 데이터
rgb_img = np.random.rand(256, 256, 3)  # 예시 RGB 이미지 (256x256 크기)
pred_mask = np.random.randint(0, 2, (256, 256))  # 예시 예측 마스크
gt_mask = np.random.randint(0, 2, (256, 256))    # 예시 정답 마스크

# 시각화 실행
visualize_metrics(rgb_img, pred_mask, gt_mask)