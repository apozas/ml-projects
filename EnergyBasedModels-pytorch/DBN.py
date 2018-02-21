# Deep Belief Network pre-trained greedily via Contrastive Divergence
#
# Author: Alejandro Pozas-Kerstjens
# Requires: numpy for numerics
#           pytorch as ML framework
#           matplotlib for plots
#           tqdm for progress bar
#           imageio for output export
#           copy for deepcopy of variables
# Last modified: Feb, 2018

import imageio
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from RBM import RBM, outer_product
from torch.autograd import Variable
from torchvision import datasets
from torch.utils.data import DataLoader
from tqdm import tqdm


class DBN(object):
    def __init__(self, n_visible=6, hidden_layer_sizes=[3, 3],
                 k=5, use_gpu=True):
        
        self.k       = k
        self.use_gpu = use_gpu

        self.rbm_layers = []
        self.n_layers   = len(hidden_layer_sizes)
        
        assert self.n_layers > 0


        # Construct DBN out of RBMs
        for i in range(self.n_layers):
            if i == 0:
                input_size = n_visible
            else:
                input_size = hidden_layer_sizes[i - 1]

            rbm_layer = RBM(n_visible=input_size,
                            n_hidden=hidden_layer_sizes[i],
                            use_gpu=use_gpu,
                            k=k)
            if self.use_gpu:
                rbm_layer = rbm_layer.cuda()
            self.rbm_layers.append(rbm_layer)
        # Copy the RBMs (in particular, the parameters), so we have one copy
        # for inference and another for generation. Needed for fine-tuning
        self.inference_layers = deepcopy(self.rbm_layers)
            


    def pretrain(self, input_data, lr=0.1, max_look_ahead=15,
                 batch_size=10, test=None):
        # Pre-train the DBN as individual RBMs
        for i in range(self.n_layers):
            print('#########################################')
            print('#        Pre-training layers {}-{}        #'.format(i, i+1))
            print('#########################################')
            if i == 0:
                layer_input = input_data
            else:
                layer_input = Variable(layer_input)
                if self.use_gpu:
                    layer_input = layer_input.cuda()
                _, sample   = (self.rbm_layers[i-1]
                                   .sample_h_given_v(layer_input))
                layer_input = sample.data
                if test is not None:
                    _, test = self.rbm_layers[i-1].sample_h_given_v(test)
            
            rbm = self.rbm_layers[i]
            best_gap = np.inf
            epoch = 1
            look_ahead = 0
            while True:
                layer_loader = torch.utils.data.DataLoader(layer_input,
                                                         batch_size=batch_size,
                                                         shuffle=True)
                rbm.train(input_data=layer_loader, lr=lr, epoch=epoch)
                if test is not None:
                    validation = Variable(layer_input)[:10000]
                    if self.use_gpu:
                        validation = validation.cuda()
                        test   = test.cuda()
                    val_fe  = rbm.free_energy(validation).mean(0)
                    test_fe = rbm.free_energy(test).mean(0)
                    gap = val_fe - test_fe
                    print('Gap: ' + str(gap.data[0]))
                    if abs(gap.data[0]) < best_gap:
                        best_gap = abs(gap.data[0])
                        print('Gap is smaller than best previous, ' +
                              'saving weights...')
                        best_weights = rbm.state_dict()
                        look_ahead = 0
                look_ahead += 1
                if look_ahead == max_look_ahead:
                    print('It has been {} epochs since the last improvement. '
                         .format(str(max_look_ahead)) + 'Stopping training...')
                    break
                epoch += 1
            rbm.load_state_dict(best_weights)


    def finetune(self, input_data, lr=0.1, epochs=100, batch_size=10, k=None):
        '''This is done according to the up-down method developed in
           Hinton et al., A fast learning algorithm for deep belief nets.
           Neural Computation 18, 1527-1554 (2006)
        '''
        print('#########################################')
        print('#        Fine-tuning full model         #')
        print('#########################################')
        if k is not None:
            self.k = k
        input_loader = torch.utils.data.DataLoader(input_data,
                                                   batch_size=batch_size,
                                                   shuffle=True)
        top_RBM      = self.rbm_layers[-1]
        for epoch in range(epochs):
            for iter, batch in enumerate(tqdm(input_loader,
                                              desc='Epoch ' + str(epoch + 1))):
                sample_data = Variable(batch).float()
                if self.use_gpu:
                    sample_data = sample_data.cuda()
        
                # Perform a bottom-up pass to get wake positive phase
                # probabilities and samples, using the inference parameters.
                # We begin with the train data as samples of the visible layer
                wakepos_samples = [sample_data]
                for i, rbm in enumerate(self.inference_layers[:-1]):
                    if i == 0:
                        layer_input = sample_data
                    else:
                        layer_input = wakepos_samples[-1]
                    _, sample = rbm.sample_h_given_v(layer_input)
                    wakepos_samples.append(sample)
                _, sample = top_RBM.sample_h_given_v(wakepos_samples[-1])
                wakepos_samples.append(sample)
                # We use these parameters to get the positive phase of the top
                # layer, which is trained as a standard RBM
                pos_W_topRBM = outer_product(wakepos_samples[-2],
                                             wakepos_samples[-1])
                # Perform Gibbs sampling iterations in the top RBM
                h_sample = sample    # To initialize loop
                for _ in range(self.k):
                    _, v_sample = top_RBM.sample_v_given_h(h_sample)
                    _, h_sample = top_RBM.sample_h_given_v(v_sample)
                # Take negative phase statistics on the top RBM for CD
                neg_W_topRBM = outer_product(v_sample, h_sample)

                # Beginning at the end of the Gibbs sampling, perform a
                # top-down generative pass to obtain the sleep positive phase
                # samples
                sleeppos_samples = [h_sample, v_sample]
                for i, rbm in reversed(list(enumerate(self.rbm_layers[:-1]))):
                    if i == self.n_layers - 1:
                        layer_input = v_sample
                    else:
                        layer_input = sleeppos_samples[-1]
                    _, sleeppos_sample = rbm.sample_v_given_h(layer_input)
                    sleeppos_samples.append(sleeppos_sample)
                # Go back to normal order, where the first element corresponds
                # to the visible layer
                sleeppos_samples = list(reversed(sleeppos_samples))
                # Predictions on the current states of the layers, for
                # the negative contributions of the wake and sleep phases.
                # Note we use probabilities instead of samples, as appears in
                # the original algorithm
                sleepneg_means = [None]  # For having even indices. Unimportant
                wakeneg_means  = []
                for i in range(self.n_layers - 1):
                    sleepneg_mean, _ = (
                                     self.inference_layers[i]               
                                         .sample_h_given_v(sleeppos_samples[i])
                                        )
                    sleepneg_means.append(sleepneg_mean)
                
                    wakeneg_mean, _ = (
                                    self.rbm_layers[i]                       
                                        .sample_v_given_h(wakepos_samples[i+1])
                                      )
                    wakeneg_means.append(wakeneg_mean)
                # Updates to generative parameters. The last layer still acts
                # as a standard RBM and we update it separately
                for i, rbm in enumerate(self.rbm_layers[:-1]):
                    wakediff_i = wakepos_samples[i] - wakeneg_means[i]
                    deltaW     = outer_product(wakepos_samples[i+1],
                                               wakediff_i).data.mean(0)
                    deltav     = wakediff_i.data.mean(0)
                    rbm.W.data     += lr * deltaW
                    rbm.vbias.data += lr * deltav
                # Updates to top RBM parameters
                deltaW = (pos_W_topRBM - neg_W_topRBM)
                deltav = (wakepos_samples[-2] - sleeppos_samples[-2])
                deltah = (wakepos_samples[-1] - sleeppos_samples[-1])
                top_RBM.W.data     += lr * deltaW.data.mean(0)
                top_RBM.vbias.data += lr * deltav.data.mean(0)
                top_RBM.hbias.data += lr * deltah.data.mean(0)
                # Updates to inference parameters
                for i, rbm in enumerate(self.inference_layers[:-1]):
                    sleepdiff_i = sleeppos_samples[i+1] - sleepneg_means[i+1]
                    deltaW      = outer_product(sleeppos_samples[i],
                                                sleepdiff_i).data.mean(0)
                    deltah      = sleepdiff_i.data.mean(0)
                    rbm.W.data     += lr * deltaW
                    rbm.hbias.data += lr * deltah


    def generate(self, k=None):
        if k is None:
            k = self.k
        rbm    = self.rbm_layers[-1]
        sample = Variable(torch.zeros(rbm.vbias.size()))
        if self.use_gpu:
            sample = sample.cuda()
        for _ in range(k):
            _, top    = rbm.sample_h_given_v(sample)
            _, sample = rbm.sample_v_given_h(top)
        for rbm_layer in reversed(self.rbm_layers[:-1]):
            _, sample = rbm_layer.sample_v_given_h(sample)
        return sample


    def save_model(self, filename):
        dicts = []
        for layer in self.rbm_layers + self.inference_layers:
            dicts.append(layer.state_dict())
        torch.save(dicts, filename)


    def load_model(self, filename):
        dicts = torch.load(filename)
        for i, layer in enumerate(self.rbm_layers + self.inference_layers):
            layer.load_state_dict(dicts[i])


def test_dbn(pretrain_lr=1e-2, look_ahead_pretrain=20, k=5, finetune_lr=1e-3,
             finetune_epochs=30, batch_size=20, use_gpu=True):

    data = datasets.MNIST('mnist',
                          train=True,
                          download=True).train_data.type(torch.FloatTensor)
    test = datasets.MNIST('mnist',
                          train=False).test_data.type(torch.FloatTensor)
    
    data = data.view((-1, 784)) / 255
    test = test.view((-1, 784)) / 255
    
    vis  = len(data[0])
    
    # -------------------------------------------------------------------------
    # Construct DBN
    # -------------------------------------------------------------------------
    pre_trained = os.path.isfile('DBN.h5')
    dbn         = DBN(n_visible=vis,
                      hidden_layer_sizes=[30, 30],
                      k=k,
                      use_gpu=use_gpu)
    if pre_trained:
        dbn.load_model('DBN.h5')
    
    if use_gpu:
        for layer in dbn.rbm_layers:
            layer = layer.cuda()
    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------
    if not pre_trained:
        test       = Variable(test)
        if use_gpu:
            test       = test.cuda()
        dbn.pretrain(input_data=data,
                     lr=pretrain_lr,
                     max_look_ahead=look_ahead_pretrain,
                     batch_size=batch_size,
                     test=test)
        dbn.finetune(input_data=data,
                     lr=finetune_lr,
                     epochs=finetune_epochs,
                     batch_size=batch_size)
    
        dbn.save_model('DBN.h5')

    # -------------------------------------------------------------------------
    # Plotting
    # -------------------------------------------------------------------------
    print('#########################################')
    print('#          Generating samples           #')
    print('#########################################')
    top_RBM = dbn.rbm_layers[-1]
    plt.figure(figsize=(20, 10))
    zero = Variable(torch.zeros(25, len(top_RBM.vbias)))
    if use_gpu:
        zero = zero.cuda()
    images = [np.zeros((5 * 28, 5 * 28))]
    for i in range(200):
        _, zero = top_RBM.sample_h_given_v(zero)
        _, zero = top_RBM.sample_v_given_h(zero)
        sample = zero
        for rbm_layer in reversed(dbn.rbm_layers[1:-1]):
            _, sample = rbm_layer.sample_v_given_h(sample)
        sample, _ = dbn.rbm_layers[0].sample_v_given_h(sample)
        datas = sample.data.cpu().numpy().reshape((25, 28, 28))
        image = np.zeros((5 * 28, 5 * 28))
        for k in range(5):
            for l in range(5):
                image[28*k:28*(k+1), 28*l:28*(l+1)] = datas[k + 5*l, :, :]
        images.append(image)
    imageio.mimsave('DBN_sample.gif', images, duration=0.1)
                                   
if __name__ == "__main__":
    test_dbn(pretrain_lr=1e-2, look_ahead_pretrain=20, k=2, finetune_lr=1e-4,
             finetune_epochs=10, batch_size=20, use_gpu=True)