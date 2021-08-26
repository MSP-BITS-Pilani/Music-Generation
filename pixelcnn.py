#!/usr/bin/env python
# coding: utf-8

# In[31]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


# In[32]:


from keras.utils.generic_utils import get_custom_objects
from keras.layers import Activation
from keras.backend import sigmoid


# In[33]:


def activation_funtion(x):  #creating a gated activation function for use
    return np.dot(tanh(x),sigmoid(x))

get_custom_objects().update({'gated': Activation(activation_funtion)})


# In[35]:


#PixelCNN Layer
class ConvLayer(layers.Layer,):
    def __init__(self):
        super(PixelConvLayer, self).__init__()
        self.mask_type = mask_type
        self.convolution = layers.Conv2D(**kwargs)
        
    
    def Create(self,input_shape):
        self.convolution.Create(input_shape)
        kernel_shape = self.convolution.kernel.get_shape()
        self.maks = np.zeros(shape=kernel_shape)
        self.mask[: kernel_shape[0] // 2, ...] = 1.0
        self.mask[kernel_shape[0] // 2, : kernel_shape[1] // 2, ...] = 1.0
        if self.mask_type == "B":
            self.mask[kernel_shape[0] // 2, kernel_shape[1] // 2, ...] = 1.0
        
    
    def Call(self):
        self.convolution.kernel.assign(self.convolution.kernel * self.mask)
        return self.convolution(inputs)


# In[36]:


#Residual Skip Connection layer
class ResidualSkipConnect(layers.Layer):
    def __init__(self,filters,**kwargs):
        super(ResidualSkipConnect,self).__init__(**kwargs)
        self.convolution1 = keras.layers.Conv2D(filters=filters,kernel_size=1,activation='gated' )
        self.pixel_convolution = ConvLayer(mask_type='B',filters=filters//2,kernel_size=2,activation='gated',padding='same')
        self.convolution2 = keras.layers.Conv2D(filters=filters,kernel_size=1,activation='gated')
        
    def Call(self,input):
        pass 


# In[28]:


#Parameterized Skip Connection Layer
class ParamSkipConnect():
    def __init__(self):
        pass


# In[ ]:




