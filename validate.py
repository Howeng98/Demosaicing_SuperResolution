from keras.models import load_model
from keras.layers import Lambda
from keras.preprocessing import image
import keras
import tensorflow as tf 
import numpy as np
#import matplotlib.pyplot as plt
from PIL import Image 
from keras import backend as K
import os
import math
oti = 'adam'
lr = 0.0001

def get_cpsnr(RGB1,RGB2,b):
	RGB1 = RGB1.astype('double'); 
	RGB2 = RGB2.astype('double');
	diff = RGB1[b:-1-b,b:-1-b,:]-RGB2[b:-1-b,b:-1-b,:];
	num = np.size(diff[:,:,1]);
	MSE_R = np.sum( np.power(diff[:,:,2],2) );
	MSE_G = np.sum( np.power(diff[:,:,1],2) );
	MSE_B = np.sum( np.power(diff[:,:,0],2) );
	CMSE = (MSE_R + MSE_G + MSE_B)/(3*num);
	CPSNR = 10*math.log(255*255/CMSE,10);
	return CPSNR;

def create_model():
  inputs = keras.Input(shape=(None,None,1))

  ##Subpixel Construction
  sub_layer_2 = Lambda(lambda x:tf.nn.space_to_depth(x,2)) 
  init = sub_layer_2(inputs=inputs)



  ##Learning Residual (DCNN)
  ####Conv 3x3x64x64 + PReLu
  x = keras.layers.Conv2D(filters = 64, #feature map number
                     kernel_size = 3, 
                     strides = 1,  # 2
                     padding = 'same', 
                     input_shape = (None,None,1))(init)
  
  x = keras.layers.PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1,2])(x)

  ####Residual Block
  for i in range(6):
    Conv1 = keras.layers.Conv2D(filters = 64, #feature map number
                       kernel_size = 3, 
                       strides = 1,  # 2
                       padding = 'same',
                       input_shape = (None,None,64))(x)
    
    PReLu = keras.layers.PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1,2])(Conv1)
    Conv2 = keras.layers.Conv2D(filters = 64, #feature map number
                       kernel_size = 3, 
                       strides = 1,  # 2
                       padding = 'same',
                       input_shape = (None,None,64))(PReLu)
   
    
    x = keras.layers.Add()([Conv2,x])
    x = keras.layers.PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1,2])(x)
  ####Conv 3x3x64x64 + PReLu
  x = keras.layers.Conv2D(filters = 64, #feature map number
                     kernel_size = 3, 
                     strides = 1,  # 2
                     padding = 'same', 
                     input_shape = (None,None,1))(x)
  
  x = keras.layers.PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1,2])(x)
  ####Conv 3x3x64x48
  x = keras.layers.Conv2D(filters = 48, #feature map number
                     kernel_size = 3, 
                     strides = 1,  
                     padding = 'same',                      
                     input_shape = (None,None,64))(x)
  
  ###########Learning Residual (DCNN)############
  

  ##Recovery From Subpixel
  sub_layer = Lambda(lambda x:tf.nn.depth_to_space(x,4)) 
  Residual_Output = sub_layer(inputs=x)
  

  ##Initial Prediction
  R = Lambda(lambda x: x[:,:,:,0])(init)
  G = Lambda(lambda x: x[:,:,:,1:3])(init)
  G = Lambda(lambda x: K.mean(x, axis=3))(G)
  B = Lambda(lambda x: x[:,:,:,3])(init)
  print(init.shape)
  print(R.shape)
  print(G.shape)
  print(B.shape)
  R = Lambda(lambda x: tf.expand_dims(x, -1))(R)
  G = Lambda(lambda x: tf.expand_dims(x, -1))(G)
  B = Lambda(lambda x: tf.expand_dims(x, -1))(B)
  
  #rgb = tf.keras.backend.stack((R, G,B),axis =  3)
  print(R.shape)
  rg = keras.layers.Concatenate(axis = 3)([R , G])
  rgb = keras.layers.Concatenate(axis = 3)([rg,B])
  print(rgb.shape)
  Coarse_Output = keras.layers.UpSampling2D(size=(4, 4))(rgb)




  ## + 
  outputs = keras.layers.Add()([Residual_Output,Coarse_Output])
 
  model = keras.Model(inputs=inputs, outputs=outputs, name="mnist_model")
  return model

model = create_model()
model.load_weights('./model.hdf5')
#model = keras.Model(inputs=(64,64,1), outputs=(128,128,3), name="mnist_model")
#model.load_weights('trashup.h5')

#model.compile(optimizer=keras.optimizers.Nadam(lr), loss = 'mean_squared_error', metrics = ['mse'])


# test_image = image.load_img('./p/im_002_0.png', target_size = (64, 64))
# test_image = image.img_to_array(test_image)
# print(test_image)
# test_image = test_image[:,:,0]
# test_image = test_image[np.newaxis,:,:,np.newaxis]
# result = model.predict(test_image)
# print(result.shape)

# out = image.array_to_img(result[0])

# plt.imshow(out)
# plt.show()

if not os.path.exists('kodaout'):
        os.makedirs('kodaout')


sum = 0
entries = os.listdir('./kodap/')
for entry in entries:
         path = './kodap/'+entry

         test_image = image.load_img(path)
         test_image = image.img_to_array(test_image)

         test_image = test_image[:,:,0]
         test_image = test_image[np.newaxis,:,:,np.newaxis]
         
         out = model.predict(test_image)
         path = './koda/'+entry
         ori = image.load_img(path)
         ori = image.img_to_array(ori)
         out = out[0];
         #out = out*255;
         print(entry,get_cpsnr(out,ori,20) );
         sum+=get_cpsnr(out,ori,20);
         out = image.array_to_img(out)
         path = './kodaout/'+entry
         out.save(path)

         # plt.imshow(out)
         # plt.show()
print('avg')        
print(sum/24)
