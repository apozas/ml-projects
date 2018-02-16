# Restricted Boltzmann Machine trained by Stochastic Gradient Descent
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
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets
from tqdm import tqdm


class RBM(nn.Module):
    def __init__(self, input_=None, n_visible=100, n_hidden=50, k=5,
                 cuda_state=True, W=None, hbias=None, vbias=None,
                 verbose=1):

        super(RBM, self).__init__()
        self.k = k
        self.input = input_
        self.cuda_state = cuda_state

        if W is not None:
            self.W = W
        else:
            # initialize W normally
            self.W = nn.Parameter(0.01 * torch.randn(n_hidden, n_visible))

        if hbias is not None:
            self.hbias = hbias
        else:
            self.hbias = nn.Parameter(torch.zeros(
                n_hidden))  # initialize h bias to 0

        if vbias is not None:
            self.vbias = vbias
        else:
            self.vbias = nn.Parameter(torch.zeros(
                n_visible))  # initialize v bias to 0
        self.verbose = verbose

    def forward(self, v):
        v0 = v
        for _ in range(self.k):
            _, h = self.sample_h_given_v(v)
            _, v = self.sample_v_given_h(h)
        return v0, v

    def free_energy(self, v):
        vbias_term = v.mv(self.vbias)
        wx_b = F.linear(v, self.W, self.hbias)

        # Now we should do log(exp(wx_b) + 1). Instead, for numerical
        # stability, we do the log-sum-exp trick, namely for large values of
        # wx_b we do a + log(exp(wx_b - a) + exp(-a)), where a = max(0, wx_b).
        zr = Variable(torch.zeros(wx_b.size()))
        if self.cuda_state:
            zr = zr.cuda()
        mask = torch.max(zr, wx_b)
        hidden_term = ((wx_b - mask).exp().add(mask.neg().exp())
                       ).log().add(mask).sum(1)
        return (-hidden_term - vbias_term).mean()

    def propdown(self, h):
        pre_sigmoid_activation = F.linear(h, self.W.t(), self.vbias)
        return F.sigmoid(pre_sigmoid_activation)

    def propup(self, v):
        pre_sigmoid_activation = F.linear(v, self.W, self.hbias)
        return F.sigmoid(pre_sigmoid_activation)

    def reconstruct(self, v, k=None):
        if self.cuda_state:
            v = v.cuda()
        if k is None:
            k = self.k
        for _ in range(k):
            _, h = self.sample_h_given_v(v)
            _, v = self.sample_v_given_h(h)
        return v

    def sample_h_given_v(self, v):
        h_mean = self.propup(v)
        h_sample = torch.bernoulli(h_mean).float()
        if self.cuda_state:
            h_sample = h_sample.cuda()
        return [h_mean, h_sample]

    def sample_v_given_h(self, h):
        v_mean = self.propdown(h)
        v_sample = torch.bernoulli(v_mean).float()
        if self.cuda_state:
            v_sample = v_sample.cuda()
        return [v_mean, v_sample]

    def train(self, input_, optimizer, epoch):
        loss_ = []
        error_ = []
        for _, batch in enumerate(tqdm(input_, desc='Epoch ' + str(epoch + 1))):
            sample_data = Variable(batch).float()
            if self.cuda_state:
                sample_data = sample_data.cuda()
            v, v1 = self.forward(sample_data)
            rec_error = F.binary_cross_entropy(v1, v)
            error_.append(rec_error.data[0])
            loss = self.free_energy(v) - self.free_energy(v1.detach())
            loss_.append(loss.data[0])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if self.verbose > 0:
            print('Loss = ' + str(np.mean(loss_)))
            print('Accuracy = ' + str(1 - np.mean(error_)))


def test_rbm(hidd=200, learning_rates=[1e-3] * 10, k=2, batch_size=30,
             cuda_state=True):

    data = datasets.MNIST('mnist', train=True,
                          download=True).train_data.type(torch.FloatTensor)
    test = datasets.MNIST(
        'mnist', train=False).test_data.type(torch.FloatTensor)

    data = data.view((-1, 784)) / 255
    data = data.bernoulli()                # Convert to binary values
    test = test.view((-1, 784)) / 255
    test = test.bernoulli()

    vis = len(data[0])
    train_loader = torch.utils.data.DataLoader(
        data, batch_size=batch_size, shuffle=True)

    # According to Hinton this should be fine, but some biases diverge in the
    # case of MNIST. Actually, this initialization is the inverse of the
    # sigmoid. This is, it is the inverse of p = sigm(vbias), so it can be
    # expected that during training the weights are close to zero and
    # change little
    vbias = nn.Parameter(
        torch.log(data.mean(0) / (1 - data.mean(0))).clamp(-20, 20))

    # ------------------------------------------------------------------------------
    # Construct RBM
    # ------------------------------------------------------------------------------
    pre_trained = os.path.isfile('RBM.h5')
    rbm = RBM(n_visible=vis, n_hidden=hidd, k=k,
              cuda_state=cuda_state, vbias=vbias)
    if pre_trained:
        rbm.load_state_dict(torch.load('RBM.h5'))

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
            # A good measure of well-fitting is the free energy difference
            # between some known and unknown instances. It is related to the
            # log-likelihood difference, but it does not depend on the
            # partition function. It should be around 0, and if it grows, it
            # might be overfitting to the training data
            gap = rbm.free_energy(validation) - rbm.free_energy(test)
            print('Gap = ' + str(gap.data[0]))
        torch.save(rbm.state_dict(), 'RBM.h5')

    # ------------------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------------------
    print('Reconstructing images')
    plt.figure(figsize=(20, 10))
    zero = Variable(torch.zeros(25, 784))
    if cuda_state:
        zero = zero.cuda()
    images = [zero.data.cpu().numpy().reshape((5 * 28, 5 * 28))]
    for i in range(2000):
        _, zero = rbm.sample_h_given_v(zero)
        _, zero = rbm.sample_v_given_h(zero)
        if i % 3 == 0:
            datas = zero.data.cpu().numpy().reshape((25, 28, 28))
            image = np.zeros((5 * 28, 5 * 28))
            for k in range(5):
                for l in range(5):
                    image[28*k:28*(k+1), 28*l:28*(l+1)] = datas[k + 5*l, :, :]
            images.append(image)
    imageio.mimsave('rbm_sample.gif', images, duration=0.1)


if __name__ == "__main__":
    learning_rates = [1e-2] * 10
    test_rbm(hidd=30, learning_rates=learning_rates,
             k=2, batch_size=20, cuda_state=False)
