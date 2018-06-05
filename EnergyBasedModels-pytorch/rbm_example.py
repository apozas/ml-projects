# Example of usage: Restricted Boltzmann Machine with continuous-valued outputs
#
# Author: Alejandro Pozas-Kerstjens
# Requires: numpy for numerics
#           pytorch as ML framework
#           matplotlib for plots
#           tqdm for progress bar
#           imageio for output export
# Last modified: Jun, 2018

import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torch.nn.functional as F
from samplers import PersistentContrastiveDivergence
from RBM import RBM
from torchvision import datasets
from tqdm import tqdm

#------------------------------------------------------------------------------
# Parameter choices
#------------------------------------------------------------------------------
hidd           = 30            # Number of nodes in the hidden layer
learning_rate  = 1e-3          # Learning rate
weight_decay   = 1e-4          # Weight decay
momentum       = 0.95          # Momentum
epochs         = 40            # Training epochs
k              = 10            # Steps of Contrastive Divergence 
k_reconstruct  = 2000          # Steps of iteration during generation
batch_size     = 20            # Batch size
model_dir      = 'CRBM.h5'     # Directory for saving last parameters learned
gpu            = False         # Use of GPU
verbose        = 1             # Additional information printout

#------------------------------------------------------------------------------
# Data preparation
#------------------------------------------------------------------------------

device = torch.device('cuda' if gpu else 'cpu')

data = datasets.MNIST('mnist',
                      train=True,
                      download=True).train_data.type(torch.float)
                      
test = datasets.MNIST('mnist',
                      train=False).test_data.type(torch.float)

data = (data.view((-1, 784)) / 255).to(device)
test = (test.view((-1, 784)) / 255).to(device)

vis = len(data[0])

# According to Hinton this initialization of the visible biases should be
# fine, but some biases diverge in the case of MNIST.
# Actually, this initialization is the inverse of the sigmoid. This is, it
# is the inverse of p = sigm(vbias), so it can be expected that during
# training the weights are close to zero and change little
vbias = torch.log(data.mean(0)/(1 - data.mean(0))).clamp(-20, 20)

# -----------------------------------------------------------------------------
# Construct RBM
# -----------------------------------------------------------------------------
sampler = PersistentContrastiveDivergence(k=k,
                                          hidden_activations=True,
                                          continuous_output=True)
rbm = RBM(n_visible=vis,
          n_hidden=hidd,
          sampler=sampler,
          device=device,
          vbias=vbias,
          verbose=verbose)
pre_trained = os.path.isfile(model_dir)
if pre_trained:
    rbm.load_state_dict(torch.load(model_dir))

# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
if not pre_trained:
    validation = data[:10000]
    for epoch in range(epochs):
        train_loader = torch.utils.data.DataLoader(data,
                                                   batch_size=batch_size,
                                                   shuffle=True)
        rbm.train(train_loader, learning_rate,
                  weight_decay, momentum, epoch + 1)
        # A good measure of well-fitting is the free energy difference
        # between some known and unknown instances. It is related to the
        # log-likelihood difference, but it does not depend on the
        # partition function. It should be around 0, and if it grows, it
        # might be overfitting to the training data.
        # High-probability instances have very negative free energy, so the
        # gap becoming very negative is sign of overfitting.
        gap = (rbm.free_energy(validation) - rbm.free_energy(test)).mean(0)
        print('Gap = {}'.format(gap.item()))
        
    torch.save(rbm.state_dict(), model_dir)

# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
print('Reconstructing images')
plt.figure(figsize=(20, 10))
zero = torch.zeros(25, 784).to(device)
images = [zero.cpu().numpy().reshape((5 * 28, 5 * 28))]
sampler.internal_sampling = True
for i in range(k_reconstruct):
    zero = sampler.get_h_from_v(zero, rbm.W, rbm.hbias)
    zero = sampler.get_v_from_h(zero, rbm.W, rbm.vbias)
    if i % 3 == 0:
        datas = zero.data.cpu().numpy().reshape((25, 28, 28))
        image = np.zeros((5 * 28, 5 * 28))
        for k in range(5):
            for l in range(5):
                image[28*k:28*(k+1), 28*l:28*(l+1)] = datas[k + 5*l, :, :]
        images.append(image)
sampler.internal_sampling = False
imageio.mimsave('RBM_sample.gif', images, duration=0.1)
