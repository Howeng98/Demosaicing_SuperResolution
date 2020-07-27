import cv2
import numpy as np
import math

PIXEL_MAX = 255.0

def psnr1(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100    
    return 20 * math.log10(PIXEL_MAX/math.sqrt(mse))

img1 = cv2.imread('1.png')
img2 = cv2.imread('2.png')
psnr_value = psnr1(img2,img1)
print("PSNR:",psnr_value)