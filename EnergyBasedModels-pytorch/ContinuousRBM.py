# Restricted Boltzmann Machine with continuous-valued outputs
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
from RBM import RBM
from torch.autograd import Variable
from torchvision import datasets
from tqdm import tqdm


class CRBM(RBM):
    def sample_v_given_h(self, h):
        '''This method of sampling is described in

        Bengio, Y., Lamblin, P., Popovici, D., and Larochelle, H.,
        Greedy Layer-wise Training of Deep Networks

        But it seems not to work better than keeping the probabilities of
        activation as the outputs themselves
        '''
        # a       = F.linear(h, self.W.t(), self.vbias)
        # a       += 1e-7    # For numerical stability
        # v_probs = F.sigmoid(a)
        # U       = Variable(torch.rand(a.size()))
        # zr      = Variable(torch.zeros(a.size()))
        # if a.is_cuda:
        #     U  = U.cuda()
        #     zr = zr.cuda()
        # mask     = torch.max(zr, a)
        # v_sample = torch.div(mask
        #                      + (U * (a - mask).exp()
        #                         + (1 - U) * mask.neg().exp()).log(),
        #                      a)
        # return [v_probs, v_sample]
        v_probs = self.propdown(h)
        return [None, v_probs]


def test_rbm(hidd=200, learning_rate=1e-2, weight_decay=0, momentum=0,
             max_look_ahead=100, k=2, k_reconstruct=100, batch_size=30,
             model_dir='CRBM.h5', best_dir='CRBM_best.h5', use_gpu=False,
             verbose=0, pcd=True):

    data = datasets.MNIST('mnist',
                          train=True,
                          download=True).train_data.type(torch.FloatTensor)
    test = datasets.MNIST('mnist',
                          train=False).test_data.type(torch.FloatTensor)

    data = data.view((-1, 784)) / 255
    test = test.view((-1, 784)) / 255

    vis = len(data[0])

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
    pre_trained = os.path.isfile('CRBM.h5')
    rbm = CRBM(n_visible=vis,
               n_hidden=hidd,
               k=k,
               use_gpu=use_gpu,
               vbias=vbias,
               verbose=verbose,
               persistent=pcd)
    if pre_trained:
        rbm.load_state_dict(torch.load('CRBM.h5'))

    if use_gpu:
        rbm = rbm.cuda()
    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------
    if not pre_trained:
        validation = Variable(data)[:10000]
        test = Variable(test)
        best_gap = np.inf
        look_ahead = 0
        epoch = 1
        while True:
            train_loader = torch.utils.data.DataLoader(data,
                                                       batch_size=batch_size,
                                                       shuffle=True)
            rbm.train(train_loader, learning_rate,
                      weight_decay, momentum, epoch)
            # A good measure of well-fitting is the free energy difference
            # between some known and unknown instances. It is related to the
            # log-likelihood difference, but it does not depend on the
            # partition function. It should be around 0, and if it grows, it
            # might be overfitting to the training data.
            # High-probability instances have very negative free energy, so the
            # gap becoming very negative is sign of overfitting.
            gap = (rbm.free_energy(validation) - rbm.free_energy(test)).mean(0)
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
    imageio.mimsave('CRBM_sample.gif', images, duration=0.1)


if __name__ == "__main__":
    test_rbm(hidd=30, learning_rate=1e-3, weight_decay=1e-4, momentum=0.9,
             max_look_ahead=20, k=20, k_reconstruct=2000, batch_size=20,
             model_dir='CRBM.h5', best_dir='CRBM_best.h5', use_gpu=False,
             verbose=1, pcd=True)
