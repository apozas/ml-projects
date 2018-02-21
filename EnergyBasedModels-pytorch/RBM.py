# Restricted Boltzmann Machine trained by Contrastive Divergence
#
# Author: Alejandro Pozas-Kerstjens
# Requires: numpy for numerics
#           pytorch as ML framework
#           matplotlib for plots
#           tqdm for progress bar
#           imageio for output export
# Last modified: Feb, 2018

import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm


def outer_product(vecs1, vecs2):
    '''Computes the outer product of batches of vectors
       :param vecs1: b 1-D tensors of length m
       :type vecs1: list of torch.Tensor or torch.autograd.Variable
       :param vecs2: b 1-D tensors of length n
       :type vecs2: list of torch.Tensor or torch.autograd.Variable
       
       :returns: torch.Tensor or torch.autograd.Variable of size (m, n)
       '''
    return torch.bmm(vecs1.unsqueeze(2), vecs2.unsqueeze(1))
    
    
def log1pexp(tensor):
    '''Computes pointwise log(1+exp()) for all elements in a torch tensor. The
       way of computing it without under- or overflows is through the
       log-sum-exp trick, namely computing
       
       log(1+exp(x)) = a + log(exp(-a) + exp(x-a))     with a = max(0, x)
       
       The function is adapted to be used in GPU if needed.
       :param tensor: torch.Tensor or torch.autograd.Variable of arbitrary size
       
       :returns: torch.Tensor or torch.autograd.Variable
       '''
    zr = Variable(torch.zeros(tensor.size()))
    if tensor.is_cuda:
        zr = zr.cuda()
    a = torch.max(zr, tensor)
    return a + (a.neg().exp() + (tensor - a).exp()).log()


class RBM(nn.Module):
    def __init__(self, n_visible=100, n_hidden=50, k=5,
	             use_gpu=True, W=None, hbias=None, vbias=None):
        
        super(RBM, self).__init__()
        self.k        = k
        self.use_gpu  = use_gpu
        if W is not None:
            self.W = W
        else:
            self.W = nn.Parameter(0.01 * torch.randn(n_hidden, n_visible))
        
        if hbias is not None:
            self.hbias = hbias
        else:
            self.hbias = nn.Parameter(torch.zeros(n_hidden))
        
        if vbias is not None:
            self.vbias = vbias
        else:
            self.vbias = nn.Parameter(torch.zeros(n_visible))


    def forward(self, v0):
        h0_probs, h0 = self.sample_h_given_v(v0)
        _, v         = self.sample_v_given_h(h0)
        for _ in range(self.k - 1):
            h_probs, h = self.sample_h_given_v(v)
            _ , v      = self.sample_v_given_h(h)
        return v0, h0_probs, v, h_probs
        
    
    def free_energy(self, v):
        if (self.use_gpu and not v.is_cuda):
            v = v.cuda()
        vbias_term  = v.mv(self.vbias)
        wx_b        = F.linear(v, self.W, self.hbias)
        hidden_term = log1pexp(wx_b).sum(1)
        return (-hidden_term - vbias_term)
        
        
    def propdown(self, h):
        if (self.use_gpu and not h.is_cuda):
            h = h.cuda()
        pre_sigmoid_activation = F.linear(h, self.W.t(), self.vbias)
        return F.sigmoid(pre_sigmoid_activation)


    def propup(self, v):
        if (self.use_gpu and not v.is_cuda):
            v = v.cuda()
        pre_sigmoid_activation = F.linear(v, self.W, self.hbias)
        return F.sigmoid(pre_sigmoid_activation)


    def reconstruct(self, v, k=None):
        if k == None:
            k = self.k
        for _ in range(k):
            _, h = self.sample_h_given_v(v)
            _, v = self.sample_v_given_h(h)
        return v
        
        
    def sample_h_given_v(self, v):
        h_probs  = self.propup(v)
        h_sample = h_probs.bernoulli()
        return [h_probs, h_sample]


    def sample_v_given_h(self, h):
        v_probs  = self.propdown(h)
        v_sample = v_probs.bernoulli()
        return [v_probs, v_sample]


    def train(self, input_data, lr, epoch):
        error_ = []
        for _, batch in enumerate(tqdm(input_data,
                                       desc='Epoch ' + str(epoch))):
            sample_data = Variable(batch).float()
            if self.use_gpu:
                sample_data = sample_data.cuda()
            v0, h0_probs, v1, h1_probs = self.forward(sample_data)
            rec_error = F.binary_cross_entropy(v1, v0)
            error_.append(rec_error.data[0])
            deltaW = (outer_product(h0_probs, v0) 
                      - outer_product(h1_probs, v1)).data.mean(0)
            deltah = (h0_probs - h1_probs).data.mean(0)
            deltav = (v0 - v1).data.mean(0)
            
            self.W.data     += lr * deltaW
            self.hbias.data += lr * deltah
            self.vbias.data += lr * deltav
        return np.mean(error_)

        
def test_rbm(hidd=200, learning_rate=1e-2, max_look_ahead=100, k=2, 
             k_reconstruct=100, batch_size=30, model_dir='RBM.h5', 
             best_dir='RBM_best.h5', use_gpu=True):
    
    data = datasets.MNIST('mnist',
                          train=True,
                          download=True).train_data.type(torch.FloatTensor)
    test = datasets.MNIST('mnist',
                          train=False).test_data.type(torch.FloatTensor)
    
    data = data.view((-1, 784)) / 255
    data = data.bernoulli()                # Convert to binary values
    test = test.view((-1, 784)) / 255
    test = test.bernoulli()
    
    vis  = len(data[0])
    
    # According to Hinton this initialization of the visible biases should be
    # fine, but some biases diverge in the case of MNIST.
    # Actually, this initialization is the inverse of the sigmoid. This is, it
    # is the inverse of p = sigm(vbias), so it can be expected that during
    # training the weights are close to zero and change little
    vbias = nn.Parameter(torch.log(
                                   data.mean(0) / (1 - data.mean(0))
                                   ).clamp(-20, 20))
	
    # -------------------------------------------------------------------------
    # Construct RBM
    # -------------------------------------------------------------------------
    pre_trained = os.path.isfile(model_dir)
    rbm         = RBM(n_visible=vis,
                      n_hidden=hidd,
                      k=k,
                      use_gpu=use_gpu,
                      vbias=vbias)
    if pre_trained:
        rbm.load_state_dict(torch.load(best_dir))
    
    if use_gpu:
        rbm = rbm.cuda()
    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------
    if not pre_trained:
        validation = Variable(data)[:10000]
        test       = Variable(test)
        best_gap   = np.inf
        look_ahead = 0
        epoch = 1
        while True:
            train_loader = torch.utils.data.DataLoader(data,
                                                       batch_size=batch_size,
                                                       shuffle=True)
            metrics      = rbm.train(train_loader, learning_rate, epoch)
            # A good measure of well-fitting is the free energy difference
            # between some known and unknown instances. It is related to the
            # log-likelihood difference, but it does not depend on the
            # partition function. It should be around 0, and if it grows, it
            # might be overfitting to the training data.
            # High-probability instances have very negative free energy, so the
            # gap becoming very negative is sign of overfitting.
            gap = (rbm.free_energy(validation) - rbm.free_energy(test)).mean(0)
            print('Rec. error = ' + str(metrics))
            print('Gap = {}'.format(gap.data[0]))
            if abs(gap.data[0]) < best_gap:
                best_gap = abs(gap.data[0])
                print('Gap is smaller than best previous, saving weights...')
                torch.save(rbm.state_dict(), best_dir)
                look_ahead = 0
            look_ahead += 1
            if look_ahead == max_look_ahead:
                print('It has been {} epochs since the last improvement. '
                      .format(str(max_look_ahead)) + 'Stopping training...')
                break
            epoch += 1
        torch.save(rbm.state_dict(), model_dir)
    
    # -------------------------------------------------------------------------
    # Plotting
    # -------------------------------------------------------------------------
    print('Reconstructing images')
    plt.figure(figsize=(20, 10))
    zero = Variable(torch.zeros(25, 784))
    images = [zero.data.numpy().reshape((5 * 28, 5 * 28))]
    for i in range(k_reconstruct):
        _, zero = rbm.sample_h_given_v(zero)
        _, zero = rbm.sample_v_given_h(zero)
        if i % 3 == 0:
            datas = zero.data.cpu().numpy().reshape((25, 28, 28))
            image = np.zeros((5 * 28, 5 * 28))
            for k in range(5):
                for l in range(5):
                    image[28*k:28*(k+1), 28*l:28*(l+1)] = datas[k + 5*l, :, :]
            images.append(image)
    imageio.mimsave('RBM_sample.gif', images, duration=0.1)


if __name__ == "__main__":
    test_rbm(hidd=30, learning_rate=1e-3, max_look_ahead=15, k=2,
             k_reconstruct=2000, batch_size=10, model_dir='RBM.h5',
             best_dir='RBM_best.h5', use_gpu=True)