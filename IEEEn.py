import numpy as np
from tensorflow.keras import layers
from keras.preprocessing import image
import tensorflow as tf 
from keras.models import Model,load_model
from keras.utils import to_categorical
import os
import keras
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau
from keras.layers import Lambda
from keras.optimizers import Adam
from keras import backend as K
import random
from PIL import Image 
from random import shuffle

batch_sz = 16
oti = 'adam'
lr = 0.0002
e_num = 20


#http://ethen8181.github.io/machine-learning/keras/resnet_cam/resnet_cam.html
#https://ithelp.ithome.com.tw/articles/10223034


def main():
  # train_image = np.load('train_image.npy')
  # train_label = np.load('train_label.npy')

  # test_image = np.load('test_img.npy')
  # test_label = np.load('test_lab.npy')


  train_image = []
  train_label = []

  entries = os.listdir('./patch')
  for entry in entries:
    im = image.load_img('./patch/'+entry, target_size = (64, 64))
    img = image.img_to_array(im)
    img = img[:,:,0]
    img = img[:,:,np.newaxis]
    train_image.append(img)
  train_image= np.stack(train_image)

  print(train_image.shape)# (x,128,128,1)
  np.save('train_image',train_image)
          
  entries = os.listdir('./label')
  for entry in entries:
    im = image.load_img('./label/'+entry, target_size = (128, 128))
    img = image.img_to_array(im)
    train_label.append(img)
  train_label = np.stack(train_label)

  print(train_label.shape)# (x,256,256,3)
  
  np.save('train_label',train_label)

  index = [i for i in range(train_image.shape[0])]
  shuffle(index)
  train_image = train_image[index,:,:,:];
  train_label = train_label[index,:,:,:];


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
  #outputs = Residual_Output
 
  model = keras.Model(inputs=inputs, outputs=outputs, name="JDMSR_model")
  model.summary()
  #model.compile(optimizer=keras.optimizers.Nadam(lr), loss = 'mean_squared_error', metrics = ['mse'])
  model.compile(optimizer=Adam(lr=0.0002, beta_1=0.9, beta_2=0.999, epsilon=1e-08), loss = 'mean_squared_error', metrics = ['mse'])
  
  #histories = Histories()
  checkpoint = ModelCheckpoint('./model.hdf5',verbose=1, monitor='val_loss', 
                                save_best_only=True,save_weights_only=True)
  rrp = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=1, mode='min', min_lr=0.0000002)

  history = model.fit(train_image, train_label, epochs=e_num, batch_size=batch_sz,verbose=1,
              validation_split = 0.1,callbacks=[checkpoint,rrp],shuffle = True)
  model.save("trashn.h5")
  loss, accuracy = model.evaluate(test_image,test_label)
  print(loss)
  
  
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

           out = image.array_to_img(out)
           plt.imshow(out)
           plt.show()


if __name__ == '__main__':
  main()
