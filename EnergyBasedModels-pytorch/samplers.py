# Collection of samplers for Energy-based models in pytorch
#
# Author: Alejandro Pozas-Kerstjens
# Requires: pytorch as ML framework
# Last modified: Apr, 2018

import torch
import torch.nn.functional as F
from torch.autograd import Variable

class Sampler(object):

    def __init__(self):
        """Constructor for the class.
        """
        self.backend = "cpu"

    def get_v_sample(self, v0, W, vbias, hbias):
        """Samples the visible layer from p(v)=sum_h p(v,h)
        
        Arguments:
        
            :param v0: Initial configuration of the visible nodes
            :type v0: torch.Tensor
            :param W: Weights connecting the visible and hidden layers
            :type W: torch.nn.Parameter
            :param vbias: Biases for the visible nodes
            :type vbias: torch.nn.Parameter
            :param hbias: Biases for the hidden nodes
            :type hbias: torch.nn.Parameter
            
            :returns: torch.autograd.Variable -- a sample configuration of the
                      visible nodes
        """
        return v0

    def get_h_from_v(self, v0, W, hbias):
        """Samples the hidden layer from the conditional distribution p(h|v)
        
        Arguments:
        
            :param v0: Initial configuration of the visible nodes
            :type v0: torch.Tensor
            :param W: Weights connecting the visible and hidden layers
            :type W: torch.nn.Parameter
            :param hbias: Biases for the hidden nodes
            :type hbias: torch.nn.Parameter
            
            :returns: torch.autograd.Variable -- a sample configuration of the
                      hidden nodes
        """
        return v0

    def get_v_from_h(self, h0, W, vbias):
        """Samples the visible layer from the conditional distribution p(v|h)
        
        Arguments:
        
            :param h0: Initial configuration of the hidden nodes
            :type h0: torch.Tensor
            :param W: Weights connecting the visible and hidden layers
            :type W: torch.nn.Parameter
            :param vbias: Biases for the visible nodes
            :type vbias: torch.nn.Parameter
            
            :returns: torch.autograd.Variable -- a sample configuration of the
                      hidden nodes
        """
        return h0

class ContrastiveDivergence(Sampler):

    def __init__(self, k, gpu=False, hidden_activations=False,
                 continuous_output=False):
        """Constructor for the class.
        
        Arguments:
        
            :param k: The number of iterations in CD-k
            :type k: int
            :param gpu: Optional parameter to indicate GPU use.
            :type gpu: bool
            :param hidden_activations: Optional parameter to output hidden
                                       activations instead of samples (claimed
                                       to improve learning)
            :type hidden_activations: bool
            :param continuous_output: Optional parameter to output visible
                                        activations instead of samples (for
                                        continuous-valued outputs)
            :type continuous_output: bool
        """
        super(ContrastiveDivergence, self).__init__()
        assert k > 0, 'You should specify a number of Gibbs steps > 0'
        self.k = k
        self.internal_sampling = False
        self.gpu = gpu
        if self.gpu:
            self.backend = "gpu"
        self.hidden_activations = hidden_activations
        self.continuous_output = continuous_output

    def get_v_sample(self, v0, W, vbias, hbias):
        """Obtains a sample of the visible layer after k steps of Contrastive
        Divergence. The method reads
        
        get h0 from p(h|v0) -> get v1 from p(v|h0) -> get h1 from p(h|v1) ...
        
        Arguments:
        
            :param v0: Initial configuration of the visible nodes
            :type v0: torch.Tensor
            :param W: Weights connecting the visible and hidden layers
            :type W: torch.nn.Parameter
            :param vbias: Biases for the visible nodes
            :type vbias: torch.nn.Parameter
            :param hbias: Biases for the hidden nodes
            :type hbias: torch.nn.Parameter
            
            :returns: torch.autograd.Variable -- a sample configuration of the
                      visible nodes
        """
        v = None
        self.internal_sampling = True
        for _ in range(self.k):
            if v is None:
                h = self.get_h_from_v(v0, W, hbias)
            else:
                h = self.get_h_from_v(v, W, hbias)
            v = self.get_v_from_h(h, W, vbias)
        self.internal_sampling = False
        return v

    def get_h_from_v(self, v, W, hbias):
        """Samples the hidden layer from the conditional distribution p(h|v)
        
        Arguments:
        
            :param v0: Initial configuration of the visible nodes
            :type v0: torch.Tensor
            :param W: Weights connecting the visible and hidden layers
            :type W: torch.nn.Parameter
            :param hbias: Biases for the hidden nodes
            :type hbias: torch.nn.Parameter
            
            :returns: torch.autograd.Variable -- a sample configuration of the
                      hidden nodes
        """
        h_probs = self._propup(v, W, hbias)
        h_sample = h_probs.bernoulli()
        return (h_probs if (self.hidden_activations
                            and not self.internal_sampling)
                else h_sample)

    def get_v_from_h(self, h, W, vbias):
        """Samples the visible layer from the conditional distribution p(v|h)
        
        Arguments:
        
            :param h0: Initial configuration of the hidden nodes
            :type h0: torch.Tensor
            :param W: Weights connecting the visible and hidden layers
            :type W: torch.nn.Parameter
            :param vbias: Biases for the visible nodes
            :type vbias: torch.nn.Parameter
            
            :returns: torch.autograd.Variable -- a sample configuration of the
                      hidden nodes
        """
        v_probs = self._propdown(h, W, vbias)
        v_sample = v_probs.bernoulli()
        return v_probs if self.continuous_output else v_sample

    def _propdown(self, h, W, vbias):
        if ((self.backend == "gpu") and not h.is_cuda):
            h = h.cuda()
        pre_sigmoid_activation = F.linear(h, W.t(), vbias)
        return F.sigmoid(pre_sigmoid_activation) 

    def _propup(self, v, W, hbias):
        if ((self.backend == "gpu") and not v.is_cuda):
            v = v.cuda()
        pre_sigmoid_activation = F.linear(v, W, hbias)
        return F.sigmoid(pre_sigmoid_activation)

class PersistentContrastiveDivergence(ContrastiveDivergence):

    def __init__(self, k, gpu=False, hidden_activations=False,
                 continuous_output=False):
        """Constructor for the class.
        
        Arguments:
        
            :param k: The number of iterations in PCD-k
            :type k: int
            :param batch_size: The batch size defines the number of
                               Markov chains
            :type batch_size: int
            :param gpu: Optional parameter to indicate GPU use.
            :type gpu: bool
        """
        super().__init__(k, gpu, hidden_activations, continuous_output)
        self.first_call = True
        
    def get_v_sample(self, v0, W, vbias, hbias):
        """Obtains a sample of the visible layer after k steps of Persistent
        Contrastive Divergence.
        
        Arguments:
        
            :param v0: Unused parameter, needed for consistency
            :type v0: None
            :param W: Weights connecting the visible and hidden layers
            :type W: torch.nn.Parameter
            :param vbias: Biases for the visible nodes
            :type vbias: torch.nn.Parameter
            :param hbias: Biases for the hidden nodes
            :type hbias: torch.nn.Parameter
            
            :returns: torch.autograd.Variable -- a sample configuration of the
                      visible nodes
        """
        if self.first_call:
            self.markov_chains = Variable(torch.rand((v0.size(0), W.size(1))))
            if self.backend == "gpu":
                self.markov_chains = self.markov_chains.cuda()
            self.first_call = False
        self.internal_sampling = True
        for _ in range(self.k):
            h = self.get_h_from_v(self.markov_chains, W, hbias)
            v = self.get_v_from_h(h, W, vbias)
            self.markov_chains.data = v.data
        self.internal_sampling = False
        return v
