import os
from PIL import Image
import numpy as np
import cv2

def calculate_psnr(original, contrast):
    mse = np.mean((original - contrast) ** 2)
    if mse == 0:
        return 100
    max_value = 255.0
    return 20 * math.log10(max_value / math.sqrt(mse))

def bayer_reverse(img):
    height,width,c = img.shape;
    tmp = np.zeros([height,width]);
    for i in range( height ):
        for j in range( width ):
            if i % 2 == 0 :
                if j % 2 == 0:
                    tmp[i][j] = img[i][j][0];#R
                else:
                    tmp[i][j] = img[i][j][1];#G
            else :
                if j % 2 == 0:
                    tmp[i][j] = img[i][j][1];#G
                else:
                    tmp[i][j] = img[i][j][2];#B

    return tmp;


dataset_path = 'koda/kodim02.png'    
img = Image.open(dataset_path)
# img.show()
img = np.array(img)
original = img

print(img.shape)
b = img[:,:,0]
g = img[:,:,1]
r = img[:,:,2]
print(b.shape)
cv2.imshow("Blue",b)
cv2.imshow("Green",g)
cv2.imshow("Red",r)

print(img[130,101,0])
print(img[130,101,1])
print(img[130,101,2])
print(img[257,91,0])
print(img[257,91,1])
print(img[257,91,2])
tmp = np.zeros([img.shape[0], img.shape[1]])
tmp = bayer_reverse(original)
print(tmp[257,91])
print(tmp.shape)

cv2.imshow('new_image',tmp)
cv2.waitKey(0)
