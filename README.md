# Demosaicing_SuperResolution

## Testing Result

1. 2021/04/16 在固定各種參數后，以Set5為基準測資
   -  使用bilinear的結果: ``PSNR:28.73``, ``SSIM:0.753``
   -  使用bicubic的結果 ：``PSNR:28.58``, ``SSIM:0.744``

2. 2021/04/16 把patch_size從64改成32，label_size從128改成64
   - ``PSNR:28.65``, ``SSIM:0.757``

3. 2021/04/16 把coarse_output拿掉
   - ``PSNR:28.98``, ``SSIM:0.787``

4. 加入reduce_lr callbacks
   - ``PSNR:28.72``, ``SSIM:0.724``
