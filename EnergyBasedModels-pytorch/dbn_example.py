# Example of usage: Deep Belief Network
#
# Author: Alejandro Pozas-Kerstjens
# Requires: numpy for numerics
#           pytorch as ML framework
#           matplotlib for plots
#           imageio for output export
# Last modified: Apr, 2018

import imageio
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from RBM import outer_product
from DBN import DBN
from samplers import ContrastiveDivergence
from torch.autograd import Variable
from torchvision import datasets
from torch.utils.data import DataLoader

#------------------------------------------------------------------------------
# Parameter choices
#------------------------------------------------------------------------------
hidden_layers   = [30, 30]    # Number of nodes on each hidden layer
pretrain_lr     = 1e-2        # Learning rate for pre-training
weight_decay    = 1e-4        # Weight decay for pre-training
momentum        = 0.95        # Momentum for pre-training
pretrain_epochs = 20          # Pre-training epochs
k               = 5           # Steps of contrastive divergence in pre-training
finetune_lr     = 1e-4        # Learning rate for fine-tuning
finetune_epochs = 30          # Finetuning epochs
batch_size      = 20          # Batch size
gpu             = False       # Use GPU
continuous_out  = True        # Whether we want continuous outputs or not
sample_copies   = 5           # Number of samples taken from the hidden
                              # representation of each datapoint

data = datasets.MNIST('mnist',
                      train=True,
                      download=True).train_data.type(torch.FloatTensor)
test = datasets.MNIST('mnist',
                      train=False).test_data.type(torch.FloatTensor)

data = data.view((-1, 784)) / 255
test = test.view((-1, 784)) / 255

vis  = len(data[0])

# -----------------------------------------------------------------------------
# Construct DBN
# -----------------------------------------------------------------------------
pre_trained = os.path.isfile('DBN.h5')

sampler = ContrastiveDivergence(k=k,
                                gpu=gpu,
                                hidden_activations=True)
dbn     = DBN(n_visible=vis,
              hidden_layer_sizes=hidden_layers,
              sample_copies=sample_copies,
              sampler=sampler,
              continuous_output=continuous_out,
              gpu=gpu)
if pre_trained:
    dbn.load_model('DBN.h5')

if gpu:
    for layer in dbn.gen_layers:
        layer = layer.cuda()
# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
if not pre_trained:
    test = Variable(test)
    if gpu:
        test = test.cuda()
    dbn.pretrain(input_data=data,
                 lr=pretrain_lr,
                 weight_decay=weight_decay,
                 momentum=momentum,
                 epochs=pretrain_epochs,
                 batch_size=batch_size,
                 test=test)
    dbn.finetune(input_data=data,
                 lr=finetune_lr,
                 epochs=finetune_epochs,
                 batch_size=batch_size)

    dbn.save_model('DBN.h5')

# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
print('#########################################')
print('#          Generating samples           #')
print('#########################################')
top_RBM = dbn.gen_layers[-1]
plt.figure(figsize=(20, 10))
zero = Variable(torch.zeros(25, len(top_RBM.vbias)))
if gpu:
    zero = zero.cuda()
images = [np.zeros((5 * 28, 5 * 28))]
for i in range(200):
    zero = sampler.get_h_from_v(zero, top_RBM.W, top_RBM.hbias)
    zero = sampler.get_v_from_h(zero, top_RBM.W, top_RBM.vbias)
    sample = zero
    for gen_layer in reversed(dbn.gen_layers[1:-1]):
        sample = sampler.get_v_from_h(sample, gen_layer.W, gen_layer.vbias)
    sampler.continuous_output = continuous_out
    sample = sampler.get_v_from_h(sample,
                                  dbn.gen_layers[0].W,
                                  dbn.gen_layers[0].vbias)
    datas = sample.data.cpu().numpy().reshape((25, 28, 28))
    image = np.zeros((5 * 28, 5 * 28))
    for k in range(5):
        for l in range(5):
            image[28*k:28*(k+1), 28*l:28*(l+1)] = datas[k + 5*l, :, :]
    images.append(image)
imageio.mimsave('DBN_sample.gif', images, duration=0.1)