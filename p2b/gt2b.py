import numpy as np
from PIL import Image
import cv2

def binary_convert(image_path):
    img_np = np.array(Image.open(image_path))
    img_np[img_np != 255] = 0 # binary convert
    return img_np

def display_image(img):
    cv2.imshow("{} img".format(img), img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    image_path = 'orfd/gt_img.png' 
    binary_img = binary_convert(image_path)
    display_image(binary_img)
    
    # save img
    # Image.fromarray(binary_img).save('binary_gt_img.png')