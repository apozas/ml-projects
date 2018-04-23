import matplotlib as mpl
mpl.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from time import time
from RBM import RBM, outer_product
from copy import deepcopy
import imageio
from ContinuousRBM import one_hot
from DBN import DBN
class CondDBN(DBN):
    def __init__(self, n_visible=6, n_labels=6, hidden_layer_sizes=[3, 3],
                 k=5, persistent=False, use_gpu=False):
        
        self.k       = k
        self.use_gpu = use_gpu

        self.n_layers   = len(hidden_layer_sizes)
        
        assert self.n_layers > 0
        # First, create the DBN instance for all but the top layer
        DBN.__init__(self, n_visible, hidden_layer_sizes, k, persistent, use_gpu)
        
        # Copy the RBMs (in particular, the parameters), so we have one copy
        # for inference and another for generation. Needed for fine-tuning
        self.inference_layers = deepcopy(self.rbm_layers)
        
        # Build label-top RBM
        self.label_layer = RBM(n_visible=n_labels,
                               n_hidden=hidden_layer_sizes[-1],
                               use_gpu=use_gpu,
                               k=k,
                               persistent=persistent)
        if self.use_gpu:
            self.label_layer = self.label_layer.cuda()


    def top_free_energy(self, layer, label):
        '''Compute free energy for the combined top-label RBM model
        '''
        label_rbm = self.label_layer
        top_rbm   = self.rbm_layers[-1]
        
        vbias_term_lab = label.mv(label_rbm.vbias)
        vbias_term_top = layer.mv(top_rbm.vbias)
        wx_b           = F.linear(label, label_rbm.W) \
                         + F.linear(layer, top_rbm.W) \
                         + top_rbm.hbias

        # Now we should do log(exp(wx_b) + 1). Instead, for numerical stability,
        # we do the log-sum-exp trick, namely for large values of wx_b we do
        # log(exp(wx_b) + 1) = a + log(exp(wx_b - a) + exp(-a)),
        # where
        # a = max(0, wx_b).
        zr = Variable(torch.zeros(wx_b.size()))
        if self.use_gpu:
            zr = zr.cuda()
        mask        = torch.max(zr, wx_b)
        hidden_term = (((wx_b - mask).exp().add(mask.neg().exp())).log()
                                                                  .add(mask)
                                                                  .sum(1))
        return (-hidden_term - vbias_term_lab - vbias_term_top).mean()
    
    
    def pretrain(self, input, lr=0.1, epochs=100, weight_decay=0,
                 momentum=0, batch_size=10, test=None):
        # Pre-train the DBN as individual RBMs
        for i in range(self.n_layers - 1):
            print('#########################################')
            print('#        Pre-training layers {}-{}        #'.format(i, i+1))
            print('#########################################')
            if i == 0:
                layer_input = [image for image, _ in input]
                labels      = [label for _, label in input]
                layer_input = torch.cat(layer_input, 0)
                label_input = torch.cat(labels, 0)
                if test is not None:
                    test_input = [image for image, _ in test]
                    test_input = torch.cat(test_input, 0)
                if self.use_gpu:
                    layer_input = layer_input.cuda()
                    label_input = label_input.cuda()
            else:
                layer_input = Variable(layer_input)
                if self.use_gpu:
                    layer_input = layer_input.cuda()
                _, sample   = self.rbm_layers[i-1].sample_h_given_v(layer_input)
                layer_input = sample.data
                if test_input is not None:
                    test_input = Variable(test_input)
                    _, test_input = self.rbm_layers[i-1].sample_h_given_v(test_input)
                    test_input    = test_input.data
            
            rbm          = self.rbm_layers[i]
            layer_loader = torch.utils.data.DataLoader(layer_input,
                                                       batch_size=batch_size,
                                                       shuffle=True)
            for epoch in range(epochs):
                rbm.train(input_data=layer_loader,
                          lr=lr,
                          weight_decay=weight_decay,
                          momentum=momentum,
                          epoch=epoch)
                if test is not None:
                    validation = Variable(layer_input)[:len(test_input), :]
                    test_set = Variable(test_input)
                    if self.use_gpu:
                        validation = validation.cuda()
                        test_set = test_set.cuda()
                    gap = (rbm.free_energy(validation)
                            - rbm.free_energy(test_set)).mean(0)
                    print('Gap: ' + str(gap.data[0]))
                
        print('#########################################')
        print('# Pre-training top and condition layers #')
        print('#########################################')
        
        label_rbm = self.label_layer
        top_rbm   = self.rbm_layers[-1]
        
        layer_input = Variable(layer_input)
        if self.use_gpu:
            layer_input = layer_input.cuda()
        _, sample   = self.rbm_layers[-2].sample_h_given_v(layer_input)
        layer_input = sample.data
        lab_top_input = [[sample, label] for sample, label in zip(layer_input, labels)]
        layer_loader = torch.utils.data.DataLoader(lab_top_input,
                                                   batch_size=batch_size,
                                                   shuffle=True)
        for epoch in range(epochs):
            for iter, batch in enumerate(tqdm(layer_loader,
                                              desc='Epoch ' + str(epoch + 1))):
                layer_sample = Variable(batch[0]).float()
                lab_sample   = Variable(batch[1]).float()
                if self.use_gpu:
                    lab_sample   = lab_sample.cuda()
                    layer_sample = layer_sample.cuda()
                    
                lab_sample_0   = lab_sample.clone()
                layer_sample_0 = layer_sample.clone()
                h_probs_0     = F.sigmoid(F.linear(lab_sample_0, label_rbm.W)
                                          + F.linear(layer_sample_0, top_rbm.W)
                                          + top_rbm.hbias)
                h_sample = torch.bernoulli(h_probs_0)
                for _ in range(self.k):
                    lab_sample = label_rbm.propdown(h_sample)
                    _, layer_sample = top_rbm.sample_v_given_h(h_sample)
                    h_probs = F.sigmoid(F.linear(lab_sample, label_rbm.W) \
                                       + F.linear(layer_sample, top_rbm.W) \
                                       + top_rbm.hbias)
                    h_sample = torch.bernoulli(h_probs)
            # Weight updates. Includes momentum and weight decay
            
            # Weight decay is only applied to W, because they are the maximum
            # responsibles for overfitting
            # Note that we multiply by the learning rate, so the function
            # optimized is (NLL - weight_decay * W)
                deltaW_top = (outer_product(h_probs_0, layer_sample_0)
                            - outer_product(h_probs, layer_sample)).data.mean(0)
                deltaW_lab = (outer_product(h_probs_0, lab_sample_0)
                            - outer_product(h_probs, lab_sample)).data.mean(0)
                deltah = (h_probs_0 - h_probs).data.mean(0)
                deltav_top = (layer_sample_0 - layer_sample).data.mean(0)
                deltav_lab = (lab_sample_0 - lab_sample).data.mean(0)

                top_rbm.W.data += lr * deltaW_top
                label_rbm.W.data += lr * deltaW_lab
                top_rbm.hbias.data += lr * deltah
                top_rbm.vbias.data += lr * deltav_top
                label_rbm.vbias.data += lr * deltav_lab

            if test is not None:
                validation = lab_top_input[:10000]
                validation_ims = [image.view((-1, len(image))) for image, _ in validation]
                validation_labs = [label.view((-1, len(label))) for _, label in validation]
                validation_ims = Variable(torch.cat(validation_ims, 0))
                validation_labs = Variable(torch.cat(validation_labs, 0))
                
                if epoch == 0:
                    test_input = Variable(test_input)
                _, test_ims = self.rbm_layers[-2].sample_h_given_v(test_input)
                test_labs = [label.view((-1, len(label))) for _, label in test]
                test_labs = Variable(torch.cat(test_labs, 0))
                if self.use_gpu:
                    validation_ims  = validation_ims.cuda()
                    validation_labs = validation_labs.cuda()
                    test_ims        = test_ims.cuda()
                    test_labs       = test_labs.cuda()
                gap = self.top_free_energy(validation_ims, validation_labs) \
                      - self.top_free_energy(test_ims, test_labs)
                print('Gap: ' + str(gap.data[0]))
                    
                    

    def finetune(self, input, lr=0.1, epochs=100, batch_size=10, k=None):
        '''This is done according to the up-down method developed in
           Hinton et al., A fast learning algorithm for deep belief nets.
           Neural Computation 18, 1527-1554 (2006)
        '''
        print('#########################################')
        print('#        Fine-tuning full model         #')
        print('#########################################')
        for gen, inf in zip(self.rbm_layers, self.inference_layers):
            inf.W.data = deepcopy(gen.W.data)
            inf.vbias.data = deepcopy(gen.vbias.data)
            inf.hbias.data = deepcopy(gen.hbias.data)
        if k is not None:
            self.k = k
        input_loader = torch.utils.data.DataLoader(input,
                                                   batch_size=batch_size,
                                                   shuffle=True)
        label_rbm = self.label_layer
        top_rbm   = self.rbm_layers[-1]
        for epoch in range(epochs):
            for iter, batch in enumerate(tqdm(input_loader,
                                              desc='Epoch ' + str(epoch + 1))):
                vis_sample = Variable(batch[0].view((-1, batch[0][0].size(1)))).float()
                lab_sample   = Variable(batch[1]).float()
                if self.use_gpu:
                    lab_sample = lab_sample.cuda()
                    vis_sample = vis_sample.cuda()
        
                # Perform a bottom-up pass to get wake positive phase probabilities
                # and samples, using the inference parameters
                # We begin with the training data as samples of the visible layer
                wakepos_samples = [vis_sample]
                for i, rbm in enumerate(self.inference_layers[:-1]):
                    if i == 0:
                        layer_input = vis_sample
                    else:
                        layer_input = wakepos_samples[-1]
                    _, sample = rbm.sample_h_given_v(layer_input)
                    wakepos_samples.append(sample)
                # Compute the state of the top layer from the previous + labels     
                topmean   = F.sigmoid(F.linear(wakepos_samples[-1], top_rbm.W)
                                    + F.linear(lab_sample, label_rbm.W)
                                    + top_rbm.hbias)
                topsample = torch.bernoulli(topmean)
                wakepos_samples.append(topsample)
                # We use these parameters also to get the positive phase of the top
                # layer, and the label layer, which are standard RBMs
                pos_W_labeltop = outer_product(wakepos_samples[-1], lab_sample)
                pos_W_topRBM   = outer_product(wakepos_samples[-1],
                                               wakepos_samples[-2])
                # Perform Gibbs sampling iterations in the top RBM
                h_sample = topsample    # To initialize loop
                for _ in range(self.k):
                    _, v_sample = top_rbm.sample_v_given_h(h_sample)
                    v_labels, _ = self.label_layer.sample_v_given_h(h_sample)
                    topmean     = F.sigmoid(F.linear(v_sample, top_rbm.W)
                                            +F.linear(v_labels, label_rbm.W)
                                            + top_rbm.hbias)
                    h_sample = torch.bernoulli(topmean)
            # Take negative phase statistics on the top RBM for CD
                neg_W_labeltop = outer_product(h_sample, v_labels)
                neg_W_topRBM   = outer_product(h_sample, v_sample)

            # Beginning at the end of the Gibbs sampling, perform a top-down
            # generative pass to obtain the sleep positive phase probabilities
            # and samples
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
            # Predictions on the current states of the layers, for obtaining the
            # negative contributions for the wake and sleep phases.
            # Note we use the probabilities instead of samples, as it appears in
            # the original algorithm
                sleepneg_means = [None]   # For having even indices. Unimportant
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
            # Updates to generative parameters. The last layer still acts as a
            # standard RBM and we update it separately
                for i, rbm in enumerate(self.rbm_layers[:-1]):
                    wakediff_i = wakepos_samples[i] - wakeneg_means[i]
                    deltaW     = outer_product(wakepos_samples[i+1],
                                               wakediff_i).data.mean(0)
                    deltav     = wakediff_i.data.mean(0)
                    rbm.W.data     += lr * deltaW
                    rbm.vbias.data += lr * deltav
                # Updates to top and label RBMs parameters
                deltaW_top = (pos_W_topRBM - neg_W_topRBM)
                deltav_top = (wakepos_samples[-2] - sleeppos_samples[-2])
                deltaW_lab = (pos_W_labeltop - neg_W_labeltop)
                deltav_lab = (lab_sample - v_labels)
                deltah     = (wakepos_samples[-1] - sleeppos_samples[-1])
                top_rbm.W.data       += lr * deltaW_top.data.mean(0)
                top_rbm.vbias.data   += lr * deltav_top.data.mean(0)
                label_rbm.W.data     += lr * deltaW_lab.data.mean(0)
                label_rbm.vbias.data += lr * deltav_lab.data.mean(0)
                top_rbm.hbias.data   += lr * deltah.data.mean(0)
                # Updates to inference parameters
                for i, rbm in enumerate(self.inference_layers[:-1]):
                    sleepdiff_i = sleeppos_samples[i+1] - sleepneg_means[i+1]
                    deltaW      = outer_product(sleepdiff_i,
                                                sleeppos_samples[i]).data.mean(0)
                    deltah      = sleepdiff_i.data.mean(0)
                    rbm.W.data     += lr * deltaW
                    rbm.hbias.data += lr * deltah
    
    
    def generate(self, label=None, k=None):
        is_label_given = True if label is not None else False
        if k is None:
            k = self.k
        rbm    = self.rbm_layers[-1]
        if label is not None:
            sample = Variable(torch.rand(rbm.vbias.size()))
            label = Variable(label)
        else:
            sample = Variable(torch.rand(torch.Size([label.size(0)] + [s for s in rbm.vbias.size()])))
            label = Variable(torch.rand(self.label_layer.vbias.size()))
        if self.use_gpu:
            sample = sample.cuda()
            label = label.cuda()
        for _ in range(k):
            h_mean = F.sigmoid(F.linear(sample, rbm.W)
                               + F.linear(label, self.label_layer.W)
                               + rbm.hbias)
            h_sample = torch.bernoulli(h_mean)
            if not is_label_given:
                _, label_sample = self.label_layer.sample_v_given_h(h_sample)   # We only do this when we are not 'clamping'
            _, sample = rbm.sample_v_given_h(h_sample)
        for rbm_layer in reversed(self.rbm_layers[:-1]):
            _, sample = rbm_layer.sample_v_given_h(sample)
        return sample
        
        
    def classify(self, sample, k=None):
        if k is None:
            k = self.k
        sample = Variable(sample)
        label = Variable(torch.zeros(self.label_layer.vbias.size()))
        if self.use_gpu:
            sample = sample.cuda()
            label = label.cuda()
        for rbm in self.rbm_layers[:-1]:
            _, sample = rbm.sample_h_given_v(sample)
        top_rbm = self.rbm_layers[-1]
        for _ in range(k):
            h_mean = F.sigmoid(F.linear(sample, top_rbm.W)
                               + F.linear(label, self.label_layer.W)
                               + top_rbm.hbias)
            h_sample = torch.bernoulli(h_mean)
            _, label = self.label_layer.sample_v_given_h(h_sample)
            # In principle, we do not update the data sample (we clamp it)
        return label
        
        
    def save_model(self, filename):
        dicts = []
        for layer in self.rbm_layers + self.inference_layers + [self.label_layer]:
            dicts.append(layer.state_dict())
        torch.save(dicts, filename)
        
    
    def load_model(self, filename):
        dicts = torch.load(filename)
        for i, layer in enumerate(self.rbm_layers
                                  + self.inference_layers
                                  + [self.label_layer]):
            layer.load_state_dict(dicts[i])


def test_dbn(pretrain_lr=1e-2, pretraining_epochs=30, k=5, weight_decay=1e-4,
             momentum=0.9, finetune_lr=1e-3, finetune_epochs=30, batch_size=20,
             pcd=True, use_gpu=True):

    data = datasets.MNIST('../mnist',
                          train=True,
                          download=True,
                          transform=transforms.Compose([transforms.ToTensor()]))
    data_set = []
    for sample in data:
        image = sample[0].view((-1, 784))
        label = one_hot([sample[1]], 10)[0]
        data_set.append([image, torch.Tensor(label)])

    test = datasets.MNIST('../mnist',
                          train=False,
                          transform=transforms.Compose([transforms.ToTensor()]))
    test_set = []
    for sample in test:
        image = sample[0].view((-1, 784))
        label = one_hot([sample[1]], 10)[0]
        test_set.append([image, torch.Tensor(label)])
    
    vis  = data_set[0][0].size(1)
    lab  = len(data_set[0][1])
    # -------------------------------------------------------------------------
    # Construct DBN
    # -------------------------------------------------------------------------
    pre_trained = os.path.isfile('cDBN.h5')
    dbn         = CondDBN(n_visible=vis,
                      n_labels=lab,
                      hidden_layer_sizes=[50, 40, 30],
                      k=k,
                      persistent=pcd,
                      use_gpu=use_gpu)
    if pre_trained:
        dbn.load_model('cDBN.h5')
    
    if use_gpu:
        for layer in dbn.rbm_layers:
            layer = layer.cuda()
        dbn.label_layer = dbn.label_layer.cuda()
    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------
    if not pre_trained:
        validation = data_set[:len(test_set)]
        test       = test_set
        dbn.pretrain(input=data_set,
                     lr=pretrain_lr,
                     epochs=pretraining_epochs,
                     weight_decay=weight_decay,
                     momentum=momentum,
                     batch_size=batch_size,
                     test=test_set)
        dbn.finetune(input=data_set,
                     lr=finetune_lr,
                     epochs=finetune_epochs,
                     batch_size=batch_size)
    
        dbn.save_model('cDBN.h5')

    # -------------------------------------------------------------------------
    # Plotting
    # -------------------------------------------------------------------------
    print('#########################################')
    print('#          Generating samples           #')
    print('#########################################')
    top_rbm   = dbn.rbm_layers[-1]
    label_rbm = dbn.label_layer
    plt.figure(figsize=(20, 10))
    zero = Variable(torch.zeros(10, len(top_rbm.vbias)))
    labels = Variable(torch.zeros(10, len(label_rbm.vbias)))
    for i in range(10):
        labels[i, i] = 1
    if use_gpu:
        zero = zero.cuda()
        labels = labels.cuda()
    images = [np.zeros((28, 10 * 28))]
    for i in range(200):
        h_mean = F.sigmoid(F.linear(zero, top_rbm.W)
                           + F.linear(labels, label_rbm.W)
                           + top_rbm.hbias)
        h_sample = torch.bernoulli(h_mean)
        # label_sample = label_RBM.propdown(h_sample)   # We comment this line because we are 'clamping'
        _, zero = top_rbm.sample_v_given_h(h_sample)
        sample = zero
        for rbm_layer in reversed(dbn.rbm_layers[1:-1]):
            _, sample = rbm_layer.sample_v_given_h(sample)
        sample, _ = dbn.rbm_layers[0].sample_v_given_h(sample)
        datas = sample.data.cpu().numpy().reshape((10, 28, 28))
        image = np.zeros((28, 10 * 28))
        for k in range(10):
            image[:, 28*k:28*(k+1)] = datas[k, :, :]
        images.append(image)
    imageio.mimsave('cdbn_sample.gif', images, duration=0.5)
    for i in range(10):
        plt.imsave('cdbn_sample_{}.png'.format(i), images[20*i])
                                   
if __name__ == "__main__":
    test_dbn(pretrain_lr=1e-2, pretraining_epochs=10, k=2, weight_decay=1e-4,
             momentum=0.9, finetune_lr=1e-3, finetune_epochs=15, batch_size=20,
             pcd=False, use_gpu=False)