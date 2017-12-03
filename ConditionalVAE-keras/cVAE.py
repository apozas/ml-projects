# Conditional Variational Autoencoder in Keras
#
# Based on https://wiseodd.github.io/techblog/2016/12/17/conditional-vae/
#
# Generates MNIST numbers of one's choice, not at random as in standard VAEs
# 
# Author: Alejandro Pozas-Kerstjens
#
# Last modified: Dec, 2017

import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
import keras.backend as K
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.layers import Input, Dense, Lambda
from keras.layers.merge import Concatenate
from keras.models import Model
from keras.objectives import binary_crossentropy
from keras.utils import to_categorical



# ------------------------------------------------------------------------------
# Function definitions
# ------------------------------------------------------------------------------
def sample_z(args):
    '''Samples from a normal distribution with mean mu and variance exp(log_sigma)
	'''
    mu, log_sigma = args
    eps = K.random_normal(shape=(n_z,), mean=0., stddev=1.)
    return mu + K.exp(log_sigma / 2.) * eps

	
def vae_loss(y_true, y_pred):
    """ Calculate total_loss = reconstruction_loss + KL_divergence """
    # Reconstruction loss: binary crossentropy
    recon = binary_crossentropy(y_true, y_pred)
    # KL divergence: has closed form since the distributions are imposed to be normal
    kl = 0.5 * K.mean(K.exp(2 * log_sigma) + K.square(mu) - 1. - 2 * log_sigma, axis=-1)
    return recon + kl
	
	
# ------------------------------------------------------------------------------
# Data preparation
# ------------------------------------------------------------------------------
(x_train, y_train), _ = mnist.load_data()
x_train = x_train.reshape((x_train.shape[0], 784)) / 255
y_train = to_categorical(y_train)

n_x       = x_train.shape[1] # 784
label_dim = y_train.shape[1] # 10


# ------------------------------------------------------------------------------
# Parameter choice
# ------------------------------------------------------------------------------    
batch_size = 50   # Batch size
n_z        = 50   # Dimension of latent space
n_epochs   = 15


# ------------------------------------------------------------------------------
# Network creation
# ------------------------------------------------------------------------------
# Q(z|X) -- encoder: (X, l) -> (mu, log_sigma)
X    = Input(shape=(n_x,)) # Size of MNIST images
cond = Input(shape=(label_dim,)) # Condition input

inputs    = Concatenate(axis=1)([X, cond])
h_q       = Dense(512, activation='relu')(inputs)
mu        = Dense(n_z, activation='linear')(h_q)
log_sigma = Dense(n_z, activation='linear')(h_q)

# P(X|z) -- decoder: (mu, log_sigma) -> (z, l) -> (X)
z      = Lambda(sample_z)([mu, log_sigma])
z_cond = Concatenate(axis=1)([z, cond])

decoder_hidden = Dense(512, activation='relu')
decoder_out    = Dense(784, activation='sigmoid')

h_p     = decoder_hidden(z_cond)
outputs = decoder_out(h_p)

# Overall VAE model, for reconstruction and training
vae = Model([X, cond], outputs)

# Encoder model, to encode input into latent variable
# We use the mean as the output as it is the center point, the representative of the Gaussian
encoder = Model([X, cond], mu)

# Generator model, generate new data given latent variable z
d_cond   = Input(shape=(label_dim,))
d_z      = Input(shape=(n_z,))
d_inputs = Concatenate(axis=1)([d_z, d_cond])
d_h      = decoder_hidden(d_inputs)
d_out    = decoder_out(d_h)
decoder  = Model([d_z, d_cond], d_out)

vae.compile(optimizer='adam', loss=vae_loss)
vae.fit([x_train, y_train], x_train, batch_size=batch_size, epochs=n_epochs, validation_split=0.1)

vae.save_weights('cVAE_weights.h5')


# ------------------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------------------
plt.figure(figsize=(20, 10))
for i in range(label_dim):
    for j in range(5):
        label = to_categorical(i, label_dim).reshape(1, label_dim)
        im    = decoder.predict([np.random.uniform(-2, 2, size=(1, n_z)), label])[0].reshape((28, 28))
        plt.subplot(5, label_dim, 10*j+i+1)
        plt.axis('off')
        plt.imshow(im, cmap='Greys_r')
plt.savefig('cVAE_predictions.png')