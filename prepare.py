import numpy as np
from numpy import genfromtxt
from numpy import asarray
import math
import copy
import os
from PIL import Image 
import cv2

patch_size = 64 #input = 64x64
label_size = 128 #output = 128x128

#get RGGB bayer image
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

#split image to prepare the train set
def split(img,name):
    height,width,c = img.shape;
    print(img.shape)
    count = 0;
    for i in range(0,height,30):
        for j in range(0,width,30):
            if( i + label_size < height and j + label_size < width ):
                tmp = np.zeros([label_size,label_size,3]);
                tmp2 = np.zeros([label_size,label_size,3]);
                
                tmp = img[ i : i + label_size, j : j + label_size,:];
                
                #save splite label
                path = 'label/'+name.split('.')[0] +'_'+str(count)+'.png';
                #tmp = tmp/255
                im = Image.fromarray(tmp)
                #tmp2 = tmp
                print(tmp2.shape)
                tmp2[:,:,0] = tmp[:,:,2]
                tmp2[:,:,1] = tmp[:,:,1]                
                tmp2[:,:,2] = tmp[:,:,0]
                #im.save(path)
                cv2.imwrite(path,tmp2)

                zoom = im.resize((patch_size,patch_size)) 
                zoom2 = np.zeros([patch_size,patch_size,3]);
                gray =  np.zeros([patch_size,patch_size]);
                
                zoom = np.array(zoom)
                zoom2[:,:,0] = zoom[:,:,2]
                zoom2[:,:,1] = zoom[:,:,1]                
                zoom2[:,:,2] = zoom[:,:,0]
                
                gray = bayer_reverse(zoom2)
                #gray = gray/255
                path = 'patch/'+name.split('.')[0] +'_'+str(count)+'.png';
                im = Image.fromarray(gray)
                #im = im.convert("L")
                #im.save(path)
                cv2.imwrite(path,gray)

                count = count + 1
  



def main():
    
    if not os.path.exists('patch'):
        os.makedirs('patch')
    
    if not os.path.exists('label'):
        os.makedirs('label')
    

    entries = os.listdir('origin/')
    for entry in entries:
        print(entry)
        path = 'origin/'+entry
        img = Image.open(path)
        img = np.array(img)   
        split(img,entry)


  

if __name__ == '__main__':
    main()
