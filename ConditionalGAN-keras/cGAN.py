# Simple example of conditional GAN in Keras
# Generates MNIST numbers of one's choice, not at random as in standard GANs
# 
# Author: Alejandro Pozas-Kerstjens
#
# Requires: tqdm for progress bar
#
# Last modified: Dec, 2017
#
# Note: tricks displayed refer to those mentioned in https://github.com/soumith/ganhacks

import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.layers import LeakyReLU, Activation, Input, Dense, Concatenate
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l1_l2
from keras.utils import to_categorical
from tqdm import tqdm

def build_gan(generator, discriminator, name="gan"):
    '''Build the GAN from a generator and a discriminator'''
    yfake = Activation("linear", name="yfake")(discriminator(generator(generator.inputs)))
    yreal = Activation("linear", name="yreal")(discriminator(discriminator.inputs))
    model = Model(generator.inputs + discriminator.inputs, [yfake, yreal], name=name)
    return model
    

def disc(image_dim, label_dim, layer_dim=1024, reg=lambda: l1_l2(1e-5, 1e-5)):
    '''Discriminator network'''
    x      = (Input(shape=(image_dim,), name='discriminator_input'))
    label  = (Input(shape=(label_dim,), name='discriminator_label'))
    inputs = (Concatenate(name='input_concatenation'))([x, label])
    a = (Dense(layer_dim, name="discriminator_h1", kernel_regularizer=reg()))(inputs)
    a = (LeakyReLU(0.2))(a)
    a = (Dense(int(layer_dim / 2), name="discriminator_h2", kernel_regularizer=reg()))(a)
    a = (LeakyReLU(0.2))(a)
    a = (Dense(int(layer_dim / 4), name="discriminator_h3", kernel_regularizer=reg()))(a)
    a = (LeakyReLU(0.2))(a)
    a = (Dense(1, name="discriminator_y", kernel_regularizer=reg()))(a)
    a = (Activation('sigmoid'))(a)
    model = Model(inputs=[x, label], outputs=a, name="discriminator")
    return model
    
    
def gen(noise_dim, image_dim, label_dim, layer_dim=1024, reg=lambda: l1_l2(1e-5, 1e-5)):
    '''Generator network'''
    z      = (Input(shape=(noise_dim,), name='generator_input'))
    label  = (Input(shape=(label_dim,), name='generator_label'))
    inputs = (Concatenate(name='input_concatenation'))([z, label])
    a = (Dense(int(layer_dim / 4), name="generator_h1", kernel_regularizer=reg()))(inputs)
    a = (LeakyReLU(0.2))(a)    # Trick 5
    a = (Dense(int(layer_dim / 2), name="generator_h2", kernel_regularizer=reg()))(a)
    a = (LeakyReLU(0.2))(a)
    a = (Dense(layer_dim, name="generator_h3", kernel_regularizer=reg()))(a)
    a = (LeakyReLU(0.2))(a)
    a = (Dense(np.prod(image_dim), name="generator_x_flat", kernel_regularizer=reg()))(a)
    a = (Activation('tanh'))(a)    
    model = Model(inputs=[z, label], outputs=[a, label], name="generator")
    return model

    
def make_trainable(net, val):
    '''Changes the trainable property of a model as a whole and layer by layer'''
    net.trainable = val
    for l in net.layers:
        l.trainable = val
    

# ------------------------------------------------------------------------------
# Data preparation
# ------------------------------------------------------------------------------
(x_train, l_train), (x_test, l_test) = mnist.load_data()
x_train = np.concatenate((x_train, x_test))
l_train = np.concatenate((l_train, l_test))

# Normalization according to Trick 1
x_train = x_train.reshape(x_train.shape[0], 784)
x_train = (x_train - 127.5) / 127.5
l_train = to_categorical(l_train)


# ------------------------------------------------------------------------------
# Parameter choice
# ------------------------------------------------------------------------------    
# Dimension of noise to be fed to the generator
noise_dim = 100
# Dimension of images generated
image_dim = 28 * 28
# Dimension of labels
label_dim = 10

batch_size  = 75
num_batches = int(x_train.shape[0] / batch_size)
num_epochs  = 20

# ------------------------------------------------------------------------------
# Network creation
# ------------------------------------------------------------------------------
# Create generator ((z, l) -> (x, l))
generator = gen(noise_dim, image_dim, label_dim)
adam      = Adam(lr=0.0002, beta_1=0.5)
generator.compile(loss='binary_crossentropy', optimizer=adam)    # Trick 9

# Create discriminator ((x, l) -> y)
discriminator = disc(image_dim, label_dim)
discriminator.compile(loss='binary_crossentropy', optimizer='SGD')    # Trick 9

# Build GAN. Note how the discriminator is set to be not trainable since the beginning
make_trainable(discriminator, False)
gan = build_gan(generator, discriminator)
gan.compile(loss='binary_crossentropy', optimizer=adam)

# ------------------------------------------------------------------------------
# Training
# ------------------------------------------------------------------------------
for epoch in range(num_epochs):
    t = tqdm(range(num_batches))
    for index in t:
        # Train the discriminator. It looks like training works best if it is trained first on only
        # real data, and then only on fake data, so let's do that. This is Trick 4.
        make_trainable(discriminator, True)
        
		# Train dicriminator on real data
        batch       = np.random.randint(0, x_train.shape[0], size=batch_size)
        image_batch = x_train[batch]
        label_batch = l_train[batch]
		
		# Label smoothing. Trick 6
        y_real      = np.ones(batch_size) + 0.2 * np.random.uniform(-1, 1, size=batch_size)
        discriminator.train_on_batch([image_batch, label_batch], y_real)
        
		# Train the discriminator on fake data
        noise_batch      = np.random.normal(0, 1, (batch_size, noise_dim))    # Trick 3
        generated_images_with_labels = generator.predict([noise_batch, label_batch])
		# Label smoothing again
        y_fake = np.zeros(batch_size) + 0.2 * np.random.uniform(0, 1, size=batch_size)
        d_loss = discriminator.train_on_batch(generated_images_with_labels, y_fake)
        
		# Train the generator. We train it through the whole model. There is a very subtle point
		# here. We want to minimize the error of the discriminator, but on the other hand we want to
		# have the generator maximizing the loss of the discriminator (make him not capable of
		# distinguishing which images are real). One way to achieve this is to change the loss
		# function of the generator by some kind of "negative loss", which in practice is
		# implemented by switching the labels of the real and the fake images. Note that when
		# training the discriminator we were doing the assignment real_image->1, fake_image->0, so
        # now we will do real_image->0, fake_image->1. The order of the outputs is [fake, real],
		# as given by build_gan(). This is Trick 2.
        make_trainable(discriminator, False)
        gan_loss = gan.train_on_batch([noise_batch, label_batch, image_batch, label_batch],
									  [y_real, y_fake])
		
        t.set_description('Epoch {}/{}'.format(epoch + 1, num_epochs))
        t.set_postfix(Discriminator_loss = d_loss, GAN_loss = gan_loss)
# Save weights. Just saving the whole GAN should work as well
generator.save_weights('generator_cGAN.h5')
discriminator.save_weights('discriminator_cGAN.h5')
gan.save_weights('gan_cGAN.h5')

# ------------------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------------------
plt.figure(figsize=(20, 10))
for i in range(label_dim):
    for j in range(5):
        im = generator.predict([np.random.uniform(-1, 1, (1, noise_dim)),
	                            to_categorical(i, label_dim)])[0].reshape((28, 28))
        plt.subplot(5, label_dim, 5*j+i+1)
        plt.axis('off')
        plt.imshow(im, cmap='Greys_r')
plt.savefig('cGAN_predictions.png')