import cv2
import numpy as np
import math
from PIL import Image 
from scipy.signal import convolve2d

PIXEL_MAX = 255.0

# Calculate PSNR
def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100    
    return 20 * math.log10(PIXEL_MAX/math.sqrt(mse))

#  Calculate SSIM
def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def filter2(x, kernel, mode='same'):
    return convolve2d(x, np.rot90(kernel, 2), mode=mode)

def compute_ssim(im1, im2, k1=0.01, k2=0.03, win_size=11, L=255):

    if not im1.shape == im2.shape:
        raise ValueError("Input Images must have the same dimensions")
    if len(im1.shape) > 2:
        raise ValueError("Please input the images with 1 channel")

    M, N = im1.shape
    C1 = (k1*L)**2
    C2 = (k2*L)**2
    window = matlab_style_gauss2D(shape=(win_size,win_size), sigma=1.5)
    window = window/np.sum(np.sum(window))

    if im1.dtype == np.uint8:
        im1 = np.double(im1)
    if im2.dtype == np.uint8:
        im2 = np.double(im2)

    mu1 = filter2(im1, window, 'valid')
    mu2 = filter2(im2, window, 'valid')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = filter2(im1*im1, window, 'valid') - mu1_sq
    sigma2_sq = filter2(im2*im2, window, 'valid') - mu2_sq
    sigmal2 = filter2(im1*im2, window, 'valid') - mu1_mu2

    ssim_map = ((2*mu1_mu2+C1) * (2*sigmal2+C2)) / ((mu1_sq+mu2_sq+C1) * (sigma1_sq+sigma2_sq+C2))

    return np.mean(np.mean(ssim_map))


if __name__ == "__main__":
    img1 = cv2.imread('pic/original.png',cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('pic/contrast.png',cv2.IMREAD_GRAYSCALE)
    img3 = cv2.imread('pic/blur.png',cv2.IMREAD_GRAYSCALE)
    img4 = cv2.imread('pic/desaturated.png',cv2.IMREAD_GRAYSCALE)
    img5 = cv2.imread('pic/noise.png',cv2.IMREAD_GRAYSCALE)
    img6 = cv2.imread('pic/jpgggg.png',cv2.IMREAD_GRAYSCALE)
    
    # PSNR
    print("==================================================")   
    print("NORMAL")
    print("==================================================")
    print("PSNR:",psnr(img1,img2))
    print("PSNR:",psnr(img1,img3))
    print("PSNR:",psnr(img1,img4))
    print("PSNR:",psnr(img1,img5))
    print("PSNR:",psnr(img1,img6))

    # SSIM
    print("==================================================")        
    print("SSIM",compute_ssim(np.array(img1),np.array(img2)))
    print("SSIM",compute_ssim(np.array(img1),np.array(img3)))
    print("SSIM",compute_ssim(np.array(img1),np.array(img4)))
    print("SSIM",compute_ssim(np.array(img1),np.array(img5)))
    print("SSIM",compute_ssim(np.array(img1),np.array(img6)))
    
    
    print("==================================================")
    print("LINEAR for compress / CUBIC for enlarge")
    print("==================================================")    
    print("Before resize:")
    print("PSNR:",psnr(img1,img2))
    print("SSIM",compute_ssim(np.array(img1),np.array(img2)))


    img7 = cv2.resize(img2,None,fx=0.5,fy=0.5,interpolation = cv2.INTER_LINEAR)
    img7 = cv2.resize(img7,None,fx=2,fy=2,interpolation = cv2.INTER_CUBIC)
    print("After resize:")
    print("PSNR:",psnr(img1,img7))
    print("SSIM",compute_ssim(np.array(img1),np.array(img7)))


    img7 = cv2.resize(img7,None,fx=0.5,fy=0.5,interpolation = cv2.INTER_LINEAR)
    img7 = cv2.resize(img7,None,fx=2,fy=2,interpolation = cv2.INTER_CUBIC) 
    print("After resize2:")
    print("PSNR:",psnr(img1,img7))
    print("SSIM",compute_ssim(np.array(img1),np.array(img7)))
    
    
    img7 = cv2.resize(img7,None,fx=0.5,fy=0.5,interpolation = cv2.INTER_LINEAR)
    img7 = cv2.resize(img7,None,fx=2,fy=2,interpolation = cv2.INTER_CUBIC)    
    print("After resize3:")
    print("PSNR:",psnr(img1,img7))
    print("SSIM",compute_ssim(np.array(img1),np.array(img7)))

    # cv2.waitKey()
    # cv2.imshow("img1",img1)
    # cv2.imshow("img7",img7)

    print("==================================================")
    print("LINEAR for compress / LINEAR for enlarge")
    print("==================================================")    
    print("Before resize:")
    print("PSNR:",psnr(img1,img2))
    print("SSIM",compute_ssim(np.array(img1),np.array(img2)))


    img7 = cv2.resize(img2,None,fx=0.5,fy=0.5,interpolation = cv2.INTER_LINEAR)
    img7 = cv2.resize(img7,None,fx=2,fy=2,interpolation = cv2.INTER_LINEAR)
    print("After resize:")
    print("PSNR:",psnr(img1,img7))
    print("SSIM",compute_ssim(np.array(img1),np.array(img7)))


    img7 = cv2.resize(img7,None,fx=0.5,fy=0.5,interpolation = cv2.INTER_LINEAR)
    img7 = cv2.resize(img7,None,fx=2,fy=2,interpolation = cv2.INTER_LINEAR) 
    print("After resize2:")
    print("PSNR:",psnr(img1,img7))
    print("SSIM",compute_ssim(np.array(img1),np.array(img7)))
    
    
    img7 = cv2.resize(img7,None,fx=0.5,fy=0.5,interpolation = cv2.INTER_LINEAR)
    img7 = cv2.resize(img7,None,fx=2,fy=2,interpolation = cv2.INTER_LINEAR)    
    print("After resize3:")
    print("PSNR:",psnr(img1,img7))
    print("SSIM",compute_ssim(np.array(img1),np.array(img7)))

    print("==================================================")
    print("CUBIC for compress / CUBIC for enlarge")
    print("==================================================")    
    print("Before resize:")
    print("PSNR:",psnr(img1,img2))
    print("SSIM",compute_ssim(np.array(img1),np.array(img2)))


    img7 = cv2.resize(img2,None,fx=0.5,fy=0.5,interpolation = cv2.INTER_CUBIC)
    img7 = cv2.resize(img7,None,fx=2,fy=2,interpolation = cv2.INTER_CUBIC)
    print("After resize:")
    print("PSNR:",psnr(img1,img7))
    print("SSIM",compute_ssim(np.array(img1),np.array(img7)))


    img7 = cv2.resize(img7,None,fx=0.5,fy=0.5,interpolation = cv2.INTER_CUBIC)
    img7 = cv2.resize(img7,None,fx=2,fy=2,interpolation = cv2.INTER_CUBIC) 
    print("After resize2:")
    print("PSNR:",psnr(img1,img7))
    print("SSIM",compute_ssim(np.array(img1),np.array(img7)))
    
    
    img7 = cv2.resize(img7,None,fx=0.5,fy=0.5,interpolation = cv2.INTER_CUBIC)
    img7 = cv2.resize(img7,None,fx=2,fy=2,interpolation = cv2.INTER_CUBIC)    
    print("After resize3:")
    print("PSNR:",psnr(img1,img7))
    print("SSIM",compute_ssim(np.array(img1),np.array(img7)))
    


    ###################################################################################################################
    # Use Bicubic and Bilinear as filter to resize Low-Resolution Image
    image = Image.open('pic/original.png')
    w,h = image.size
    
    image_resize = image.resize((w/2,h/2)) #default
    image_resize1 = image.resize((w/2,h/2),Image.BILINEAR)
    image_resize2 = image.resize((w/2,h/2),Image.BICUBIC)
    image_resize3 = image.resize((w/2,h/2),Image.ANTIALIAS)

    image_resize = image.resize((w,h)) #default
    image_resize1 = image.resize((w,h),Image.BILINEAR)
    image_resize2 = image.resize((w,h),Image.BICUBIC)
    image_resize3 = image.resize((w,h),Image.ANTIALIAS)
    
    image = cv2.cvtColor(np.asarray(image) , cv2.COLOR_RGB2BGR)  
    image_resize = cv2.cvtColor(np.asarray(image_resize) , cv2.COLOR_RGB2BGR) 
    image_resize1 = cv2.cvtColor(np.asarray(image_resize1) , cv2.COLOR_RGB2BGR) 
    image_resize2 = cv2.cvtColor(np.asarray(image_resize2) , cv2.COLOR_RGB2BGR) 
    image_resize3 = cv2.cvtColor(np.asarray(image_resize3) , cv2.COLOR_RGB2BGR) 

    image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    image_resize = cv2.cvtColor(image_resize,cv2.COLOR_RGB2GRAY)
    image_resize1 = cv2.cvtColor(image_resize1,cv2.COLOR_RGB2GRAY)
    image_resize2 = cv2.cvtColor(image_resize2,cv2.COLOR_RGB2GRAY)
    image_resize3 = cv2.cvtColor(image_resize3,cv2.COLOR_RGB2GRAY)

    # PSNR
    print("============Bicubic and Bilinear==================")
    print("==================================================")
    print("PSNR:",psnr(image,image_resize))
    print("PSNR:",psnr(image,image_resize1))
    print("PSNR:",psnr(image,image_resize2))
    print("PSNR:",psnr(image,image_resize3))

    # SSIM
    print("==================================================")        
    print("SSIM",compute_ssim(np.array(image),np.array(image_resize)))
    print("SSIM",compute_ssim(np.array(image),np.array(image_resize1)))
    print("SSIM",compute_ssim(np.array(image),np.array(image_resize2)))
    print("SSIM",compute_ssim(np.array(image),np.array(image_resize3)))    
    # Image.fromarray(np.hstack((np.array(image_resize),np.array(image_resize1),np.array(image_resize2),np.array(image_resize3)))).show()

