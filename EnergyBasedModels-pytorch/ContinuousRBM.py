# Restricted Boltzmann Machine with continuous-valued outputs, trained by Stochastic Gradient Descent
#
# Author: Alejandro Pozas-Kerstjens
# Requires: numpy for numerics
#           pytorch as ML framework
#           matplotlib for plots
#           tqdm for progress bar
#           imageio for output export
# Last modified: Jan, 2018

import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from RBM import RBM
from torch.autograd import Variable
from torchvision import datasets
from tqdm import tqdm


class CRBM(RBM):
    def sample_v_given_h(self, h0_sample):
        # According to Hinton, this (using the activation probabilities as outputs themselves) should be
        # enough to get continuous outputs
        a_h = self.propdown(h0_sample)
        return [None, a_h]


def test_rbm(hidd=200, learning_rates=[1e-3] * 10, k=2, batch_size=30, cuda_state=True):

    data = datasets.MNIST('mnist', train=True,
                          download=True).train_data.type(torch.FloatTensor)
    test = datasets.MNIST(
        'mnist', train=False).test_data.type(torch.FloatTensor)

    data = data.view((-1, 784)) / 255
    test = test.view((-1, 784)) / 255

    vis = len(data[0])
    train_loader = torch.utils.data.DataLoader(
        data, batch_size=batch_size, shuffle=True)

    # According to Hinton this should be fine, but some biases diverge in the case of MNIST.
    # Actually, this initialization is the inverse of the sigmoid. This is, it is the inverse of
    # p = sigm(vbias), so it can be expected that during training the weights are close to zero and
    # change little
    vbias = nn.Parameter(
        torch.log(data.mean(0) / (1 - data.mean(0))).clamp(-20, 20))

    # ------------------------------------------------------------------------------
    # Construct CRBM
    # ------------------------------------------------------------------------------
    pre_trained = os.path.isfile('CRBM.h5')
    rbm = CRBM(n_visible=vis, n_hidden=hidd, k=k,
               cuda_state=cuda_state, vbias=vbias)
    if pre_trained:
        rbm.load_state_dict(torch.load('CRBM.h5'))

    if cuda_state:
        rbm = rbm.cuda()

    # ------------------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------------------
    if not pre_trained:
        validation = Variable(data)[:10000]
        test = Variable(test)
        if cuda_state:
            validation = validation.cuda()
            test = test.cuda()
        for epoch, learning_rate in enumerate(learning_rates):
            train_op = optim.SGD(rbm.parameters(), lr=learning_rate)
            rbm.train(train_loader, train_op, epoch)
            # A good measure of well-fitting is the free energy difference between some known and
            # unknown instances. It is related to the log-likelihood difference, but it does not
            # depend on the partition function. It should be around 0, and if it grows, it might be
            # overfitting to the training data
            gap = rbm.free_energy(validation) - rbm.free_energy(test)
            print('Gap = ' + str(gap.data[0]))
        torch.save(rbm.state_dict(), 'CRBM.h5')

    # ------------------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------------------
    print('Reconstructing images')
    plt.figure(figsize=(20, 10))
    zero = Variable(torch.zeros(25, 784))
    if cuda_state:
        zero = zero.cuda()
    images = [zero.data.cpu().numpy().reshape((5 * 28, 5 * 28))]
    for i in range(20000):
        _, zero = rbm.sample_h_given_v(zero)
        _, zero = rbm.sample_v_given_h(zero)
        if i % 32 == 0:
            datas = zero.data.cpu().numpy().reshape((25, 28, 28))
            image = np.zeros((5 * 28, 5 * 28))
            for k in range(5):
                for l in range(5):
                    image[28*k:28*(k+1), 28*l:28*(l+1)] = datas[k + 5*l, :, :]
            images.append(image)
    imageio.mimsave('crbm_sample.gif', images, duration=0.1)


if __name__ == "__main__":
    learning_rates = [1e-2] * 10
    test_rbm(hidd=30, learning_rates=learning_rates,
             k=2, batch_size=20, cuda_state=True)
