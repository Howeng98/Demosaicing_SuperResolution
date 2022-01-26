# Demosaicing_SuperResolution
![Status](https://img.shields.io/static/v1.svg?label=Status&message=Finished&color=green)

This repo is trying to make a way that can make you image reduce its [Demosaicing](https://en.wikipedia.org/wiki/Demosaicing) and [Super Resolution](https://en.wikipedia.org/wiki/Super-resolution_imaging) executing time by joint them in a model. And also enhance image resolution quality by algorithm.

## Why

An usual digital camera will use [Color Filter Array](https://en.wikipedia.org/wiki/Color_filter_array) which generated by a [CMOS](https://en.wikipedia.org/wiki/CMOS), it only can detect one color in each pixel (because if we want detect R,G,B in each pixel, we need to prepare 3 kinds of sensor in each pixel, that will make company cost a lot!), hence we will have a **R, G, G, B** ``Bayer input image``. Make the 1 dimension input image into 3 dimentions output image is called **Demosaicing**.


Except demosaic, we also need to consider image resolution. This repo is using a way called **Super Resolution** to enlarge image and enhance its resolution.


## Related Work

**1. Color Filter Array, Bayer Input Data**

![bayer_data](https://user-images.githubusercontent.com/44123278/127734437-6bdd0031-4b2f-4b82-8105-4fdcda890c37.png)

**2. Demosaicing**

![demosaicking](https://user-images.githubusercontent.com/44123278/127734451-b1ccabd3-d038-4d2c-9026-23994c463f48.png)


**3. Super Resolution**

![superresolution](https://user-images.githubusercontent.com/44123278/127734471-7ec3c5bf-3be1-4f58-9933-3420d965c8a0.png)


## Processing Pipeline

![processing_pipeline](https://user-images.githubusercontent.com/44123278/127734479-181bdfe8-fbfc-419f-882e-cce5c0e115f4.png)

## System Architecture Overview

![system_structure](https://user-images.githubusercontent.com/44123278/127734502-f58cbc40-0708-4405-bec3-f1c8f8a0f611.png)

![model_design](https://user-images.githubusercontent.com/44123278/127734509-0896c8e7-1509-492a-84ee-71415adc01a9.png)

## Training Dataset

   - **BSD200** : 200 Images with 481x321

## Testing Dataset

   - **Set5** : 5 Images
   - **Kodak** : 24 Images
   - **BSD100** : 100 Images

   ### Dataset overview
   ![set5](https://user-images.githubusercontent.com/44123278/127734697-a34eadf8-25a2-4596-99ee-804cf798bc72.png)
   
   ![kodak](https://user-images.githubusercontent.com/44123278/127734698-4a1febd9-0491-4fde-8157-946257c1acf9.png)
   
   ![bsd100](https://user-images.githubusercontent.com/44123278/127734700-0f886a98-aa5a-46a9-ad7c-4c3e89abc478.png)

## Performance Result Overview

![performence](https://user-images.githubusercontent.com/44123278/127734715-2b0e018f-c2c5-4954-a081-e5988907619c.png)

![p1](https://user-images.githubusercontent.com/44123278/127734725-2ddedfb2-eddd-47aa-a843-7c157935f77d.png)
![p2](https://user-images.githubusercontent.com/44123278/127734728-0ec137bd-0903-4983-bbf9-a88585a3e47b.png)
![p3](https://user-images.githubusercontent.com/44123278/127734731-9c8444cb-b18b-4b12-a75f-50a6eb1a6d70.png)
![p5](https://user-images.githubusercontent.com/44123278/127734732-e1bac7f8-302d-4a62-98c5-c49871df6763.png)


## Testing Log and Notes

01. 2021/04/16 在固定各種參數后，以Set5為基準測資
   -  使用bilinear的結果: ``PSNR:28.73``, ``SSIM:0.753``
   -  使用bicubic的結果 ：``PSNR:28.58``, ``SSIM:0.744``

02. 2021/04/16 把patch_size從64改成32，label_size從128改成64
   - ``PSNR:28.65``, ``SSIM:0.757``

03. 2021/04/16 把coarse_output拿掉
   - ``PSNR:28.98``, ``SSIM:0.787``

04. reduce_lr callbacks
   - 把reduce_lr加進去:``PSNR:28.72``, ``SSIM:0.724``
   - 把reduce_lr撤回去: ``PSNR:28.87``, ``SSIM:0.738``

05. 同上設定，batch_size改成16,Model_output只有Residual_Output
   - ``PSNR:29.31``, ``SSIM:0.805``

06. 64，128，30，T91，/255.0，6 residual_block, output=Residual_Output, without reduce_lr
   - ``PSNR:29.29``, ``SSIM:0.821``

07. 把koda也加入training data set中，training_dataset(koda, T91, origin)
   - original ``PSNR:29.29``, ``SSIM:0.821``
   - current  ``PSNR:27.96``, ``SSIM:0.421``

08. 2021/04/21 做bayer_reverse

09. 2021/04/21 計算RGB和BGR的PSNR

10. 2021/04/21 testing的圖加上preprocessing (bayer_reverse + resize)

11. 加入BSD200

12. 2021/04/24 創立一個縮小版本的BSD200，稱為BSD，裡面有隨機從BSD200選出30張的圖片，目的是使用輕量版的dataset測試程式碼的正確性

13. 2021/04/24 **重要！ image.load_image這個method會使到讀進來的圖片強制變成3通道！ 所以應該要用Image.open來讀取patch image**

14. 2021/04/24 稍微調整了bayer_reverse里的 R和B通道的寫法，晚點可以調回來

15. 2021/04/24 通道對調的寫法很重要 label[:,:,0] = tmp2[:,:,2]， 沒有這個部分的處理的話，藍色和紅色的色域會調換，所以不能

16. 2021/04/26 TODO, 使用bicubic

17. 2021/04/26 TODO, upsampling interpolation use 'bilinear'
   - original ``PSNR:28.82``, ``SSIM:0.774``
   - current  ``PSNR:28.95``, ``SSIM:0.784``

18. 2021/04/26 Remove rrp callbacks
   - original ``PSNR:28.95``, ``SSIM:0.784``
   - current  ``PSNR:29.69``, ``SSIM:0.787``

19. 2021/04/26 Testing kodak_HR dataset
   - current ``PSNR:29.70``, ``SSIM:0.873``

20. 2021/04/26 add 2 more Conv layers in the first layer of DCNN part
   - original ``PSNR:28.75``, ``SSIM:0.787``
   - current  ``PSNR:30.25``, ``SSIM:0.813``

21. 2021/05/03 try to figure out ``fusion`` and ``ensemble``.

22. 2021/05/13 use BSD200 as training dataset, result show that if we add more training dataset will make PSNR improve
   - current ``PSNR:31.45``, ``SSIM:0.798``

## Future Work

   1. Try to add some layers, and Dense Block into model, dense structure is good at extracting important information from Image/Video.
   2. Design your own loss function for this purpose. We are doing two great topic (**Demosaicing** and **SR**), so create a specific loss function for this model may make output Image quality upgrade.
   3. Redesign a small model for CFA pixel input data convert to 4 dimensions input image, after that link to DCNN model to continue process. (Implement own model instead of call defined function from library)
   4. Feel free to contact me if you have any problem or idea!

## References

   1. [Ruofan Zhou, Radhakrishna Achanta, Sabine Süsstrunk, “Deep Residual Network for Joint Demosaicing and
Super-Resolution,”IC EPFL, arXiv:1802.06573v1 2018.](https://asset-pdf.scinapse.io/prod/2788066301/2788066301.pdf)
   2. [Junkang Zhang, Cheolhong An, Truong Nguyen, “Deep Joint Demosaicing and Super Resolution on High
Resolution Bayer Sensor Data,”2018](https://ieeexplore.ieee.org/document/8646321)
   3. [W. T. Huang, W. J. Chen, S. C. Tai, “A Sharp Edge-Preserving Joint Color Demosaicking and Zooming
Algorithm using Integrated Gradients and An Iterative Back-Projection Technique,”2013.](https://dl.acm.org/doi/abs/10.1016/j.dsp.2013.12.007)
   4. [DengWen Zhou, WeiMing Dong, Wengang Chen, “Joint Demosaicking and Zooming Using Moderate Spectral
Correlation and Consistent Edge Map,”2014.](https://www.researchgate.net/publication/272295875_Joint_demosaicking_and_zooming_using_moderate_spectral_correlation_and_consistent_edge_map)
