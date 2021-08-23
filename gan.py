import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras import layers
import os
import numpy as np


def random_input(dimension, b_size):
    rand = np.reshape(np.random.randn(dimension * b_size), (b_size, dimension))
    
    return rand


def get_fake(model, size):
    generated = random_input(100, size)
    X_fake = model.predict(generated)
    y_fake = np.zeros((size, 1))


    return X_fake, y_fake
    
def generator():
    #Basically using SpecGAN with transposed convolutions instead of Conv + Upsampling layers, LeakyRELU instead of RELU, and Dropout
    #128 x 128 sized images are used, although the sampling rate used in SpecGAN is 16000. Can be modified
    model = keras.models.Sequential()
    model.add(layers.Dense(16384, input_dim = 100))
    model.add(layers.Reshape((4,4,1024)))
    model.add(layers.LeakyReLU(alpha = 0.2))
    model.add(layers.Conv2DTranspose(512, (5, 5), strides = (2, 2), padding = 'same'))
    model.add(layers.LeakyReLU(alpha = 0.2))
    model.add(layers.Dropout(0.4))
    model.add(layers.Conv2DTranspose(256, (5, 5), strides = (2, 2), padding = 'same'))
    model.add(layers.LeakyReLU(alpha = 0.2))
    model.add(layers.Dropout(0.4))
    model.add(layers.Conv2DTranspose(128, (5, 5), strides = (2, 2), padding = 'same'))
    model.add(layers.LeakyReLU(alpha = 0.2))
    model.add(layers.Dropout(0.4))
    model.add(layers.Conv2DTranspose(64, (5, 5), strides = (2, 2), padding = 'same'))
    model.add(layers.LeakyReLU(alpha = 0.2))
    model.add(layers.Dropout(0.4))
    model.add(layers.Conv2DTranspose(1, (5, 5), strides = (2, 2), padding= 'same'))
    model.add(layers.Activation('tanh'))

    return model

def discriminator():
    model = keras.models.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides = (2, 2), padding= 'same', input_shape = (128, 128, 1)))
    model.add(layers.LeakyReLU(alpha = 0.2))
    model.add(layers.Conv2D(128, (5, 5), strides = (2, 2), padding = 'same', kernel_initializer='he normal'))
    model.add(layers.LeakyReLU(alpha = 0.2))
    model.add(layers.Conv2D(256, (5, 5), strides = (2, 2), padding=  'same', kernel_initializer='he normal'))
    model.add(layers.LeakyReLU(alpha = 0.2))
    model.add(layers.Conv2D(512, (5, 5), strides = (2, 2), padding = 'same', kernel_initializer='he normal'))
    model.add(layers.LeakyReLU(alpha = 0.2))
    model.add(layers.Conv2D(1024, (5, 5), strides = (2, 2), padding= 'same', kernel_initializer='he normal'))
    model.add(layers.LeakyReLU(alpha = 0.2))
    model.add(layers.Flatten())
    model.add(layers.Dense(1,activation = 'sigmoid'))
    opt = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss='binary crossentropy', optimizer=opt,metrics=['accuracy'])
    

    return model


def combined(gen,disc):
    disc.trainable = False
    model = keras.models.Sequential()
    model.add(gen)
    model.add(disc)
    opt = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss= 'binary crossentropy', optimizer=opt)


    return model


def train_gan(model_gen, model_disc, directory, model_combined, epochs = 500, batch_size = 128): #To be tuned
    total_files = len([name for name in os.listdir('directory')])
    batch_number = int(total_files / batch_size)
    for i in range(epochs):
        for j in range(batch_number):
            Xr, yr = get_real(directory, batch_size)
            Xf, yf = get_fake(model_gen, batch_size)
            disc_loss_real, _ = model_disc.train_on_batch(Xr, yr)
            disc_loss_fake, _ = model_disc.train_on_batch(Xf, yf)
            disc_loss = 0.5 * (np.add(np.array(disc_loss_real), np.array(disc_loss_fake)))
            #Improvement: Try to implement Wasserstein Loss instead of this
            Xt = random_input(100, batch_size)
            yt = np.ones((batch_size, 1))
            gen_loss = model_combined.train_on_batch(Xt, yt)
            print('Epoch : %d/%d, Batch : %d/%d, DLoss: %.3f, GLoss : %.3f' % (i + 1, j + 1, disc_loss, gen_loss))
            #Need to define a function to save checkpoints
    
    return