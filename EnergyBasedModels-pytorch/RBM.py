# Restricted Boltzmann Machine
#
# Author: Alejandro Pozas-Kerstjens
# Requires: numpy for numerics
#           pytorch as ML framework
#           tqdm for progress bar
# Last modified: Jun, 2018

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Module, Parameter
from tqdm import tqdm


def outer_product(vecs1, vecs2):
    '''Computes the outer product of batches of vectors

    Arguments:

        :param vecs1: b 1-D tensors of length m
        :type vecs1: list of torch.Tensor
        :param vecs2: b 1-D tensors of length n
        :type vecs2: list of torch.Tensor
        :returns: torch.Tensor of size (m, n)
       '''
    return torch.bmm(vecs1.unsqueeze(2), vecs2.unsqueeze(1))


class RBM(Module):

    def __init__(self, n_visible=100, n_hidden=50, sampler=None, optimizer=None,
                 device=None, verbose=0, W=None, hbias=None, vbias=None):
        '''Constructor for the class.

        Arguments:

            :param n_visible: The number nodes in the visible layer
            :type n_visible: int
            :param n_hidden: The number nodes in the hidden layer
            :type n_hidden: int
            :param sampler: Method used to draw samples from the model
            :type sampler: :class:`samplers`
            :param optimizer: Optimizer used for parameter updates
            :type optimizer: :class:`optimizers`
            :param device: Device where to perform computations. None is CPU.
            :type device: torch.device
            :param verbose: Optional parameter to set verbosity mode
            :type verbose: int
            :param W: Optional parameter to specify the weights of the RBM
            :type W: torch.nn.Parameter
            :param hbias: Optional parameter to specify the hidden biases of
                          the RBM
            :type hbias: torch.nn.Parameter
            :param vbias: Optional parameter to specify the visibile biases of
                          the RBM
            :type vbias: torch.nn.Parameter
        '''

        super(RBM, self).__init__()
        self.verbose = verbose

        if device is not None:
            self.device = device
        else:
            self.device = torch.device('cpu')
        
        if W is not None:
            self.W = Parameter(W.to(self.device))
        else:
            self.W = Parameter(torch.Tensor(
                                        0.01 * torch.randn(n_hidden, n_visible)
                                            ).to(self.device))

        if hbias is not None:
            self.hbias = Parameter(hbias.to(self.device))
        else:
            self.hbias = Parameter(torch.Tensor(
                                                torch.zeros(n_hidden)
                                                ).to(self.device))

        if vbias is not None:
            self.vbias = Parameter(vbias.to(self.device))
        else:
            self.vbias = Parameter(torch.Tensor(
                                                torch.zeros(n_visible)
                                                ).to(self.device))

        if optimizer is None:
            raise Exception('You must provide an appropriate optimizer')
        self.optimizer = optimizer

        if sampler is None:
            raise Exception('You must provide an appropriate sampler')
        self.sampler = sampler

    def free_energy(self, v):
        '''Computes the free energy for a given state of the visible layer.

        Arguments:

            :param v: The state of the visible layer of the RBM
            :type v: torch.Tensor

            :returns: torch.Tensor
        '''
        vbias_term = v.mv(self.vbias)
        wx_b = F.linear(v, self.W, self.hbias)
        hidden_term = F.softplus(wx_b).sum(1)
        return (-hidden_term - vbias_term)

    def train(self, input_data):
        '''Trains the RBM.

        Arguments:

            :param input_data: Batch of training points
            :type input_data: torch.utils.data.DataLoader
            :param lr: Learning rate
            :type lr: float
            :param weight_decay: Weight decay parameter, to prevent overfitting
            :type weight_decay: float
            :param momentum: Momentum parameter, for improved learning
            :type momentum: float
            :param epoch: Optional parameter to show epoch in screen
            :type epoch: int
        '''
        error_ = []
        for batch in tqdm(input_data, desc=('Epoch ' +
                                            str(self.optimizer.epoch + 1))):
            sample_data = batch.float()
            # Sampling from the model to compute updates
            # Get positive phase from the data
            vpos = sample_data
            hpos = self.sampler.get_h_from_v(vpos, self.W, self.hbias)
            # Get negative phase from the chains
            vneg, hneg = self.sampler.get_negative_phase(vpos, self.W,
                                                        self.vbias, self.hbias)

            # Weight updates. Includes momentum and weight decay
            W_update, vbias_update, hbias_update = \
                             self.optimizer.get_updates(vpos, vneg, hpos, hneg,
                                                self.W, self.vbias, self.hbias)

            self.W.data     += W_update.data
            self.hbias.data += hbias_update.data
            self.vbias.data += vbias_update.data

        self.optimizer.epoch += 1

        if self.verbose > 0:
            rec_error = F.mse_loss(vneg, vpos)
            error_.append(rec_error.item())
            print('Reconstruction error = ' + str(np.mean(error_)))
