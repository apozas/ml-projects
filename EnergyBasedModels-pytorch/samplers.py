# Collection of samplers for Energy-based models in pytorch
#
# Author: Alejandro Pozas-Kerstjens
# Requires: pytorch as ML framework
# Last modified: Jun, 2018

from torch import rand, sigmoid
from torch.nn.functional import linear, dropout

class Sampler(object):

    def __init__(self):
        """Constructor for the class.
        """

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

            :returns: torch.Tensor -- a sample configuration of the
                      visible nodes
        """

    def get_h_from_v(self, v0, W, hbias):
        """Samples the hidden layer from the conditional distribution p(h|v)

        Arguments:

            :param v0: Initial configuration of the visible nodes
            :type v0: torch.Tensor
            :param W: Weights connecting the visible and hidden layers
            :type W: torch.nn.Parameter
            :param hbias: Biases for the hidden nodes
            :type hbias: torch.nn.Parameter

            :returns: torch.Tensor -- a sample configuration of the
                      hidden nodes
        """

    def get_v_from_h(self, h0, W, vbias):
        """Samples the visible layer from the conditional distribution p(v|h)

        Arguments:

            :param h0: Initial configuration of the hidden nodes
            :type h0: torch.Tensor
            :param W: Weights connecting the visible and hidden layers
            :type W: torch.nn.Parameter
            :param vbias: Biases for the visible nodes
            :type vbias: torch.nn.Parameter

            :returns: torch.Tensor -- a sample configuration of the
                      visible nodes
        """

    def get_negative_phase(self, v0, W, vbias, hbias):
        """Returns the negative phase for both the visible and hidden nodes

        Arguments:

            :param v0: Initial configuration of the visible nodes
            :type v0: torch.Tensor
            :param W: Weights connecting the visible and hidden layers
            :type W: torch.nn.Parameter
            :param vbias: Biases for the visible nodes
            :type vbias: torch.nn.Parameter
            :param hbias: Biases for the hidden nodes
            :type hbias: torch.nn.Parameter

            :returns: torch.Tensor -- a sample configuration of the
                      visible nodes
        """


class ContrastiveDivergence(Sampler):

    def __init__(self, k, dropout=0, hidden_activations=False,
                 continuous_output=False):
        """Constructor for the class.

        Arguments:

            :param k: The number of iterations in CD-k
            :type k: int
            :param dropout: Optional parameter, fraction of neurons in the
                            previous layer that are not taken into account when
                            getting a sample.
            :type dropout: float
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
        assert (dropout >= 0) and (dropout <= 1), ('The dropout rate' +
                                                   ' should be in [0, 1]')
        self.dropout = dropout
        self.internal_sampling = False
        self.hidden_activations = hidden_activations
        self.continuous_output = continuous_output

    def get_v_sample(self, v0, W, vbias, hbias):
        v = None
        pre_internal_sampling = self.internal_sampling
        self.internal_sampling = True
        for _ in range(self.k):
            if v is None:
                h = self.get_h_from_v(v0, W, hbias)
            else:
                h = self.get_h_from_v(v, W, hbias)
            v = self.get_v_from_h(h, W, vbias)
        self.internal_sampling = pre_internal_sampling
        return v

    def get_h_from_v(self, v, W, hbias):
        h_probs = self._propup(v, W, hbias)
        h_sample = h_probs.bernoulli()
        return (h_probs if (self.hidden_activations
                            and not self.internal_sampling)
                else h_sample)

    def get_v_from_h(self, h, W, vbias):
        v_probs = self._propdown(h, W, vbias)
        v_sample = v_probs.bernoulli()
        return v_probs if self.continuous_output else v_sample

    def get_negative_phase(self, v0, W, vbias, hbias):
        vneg = self.get_v_sample(v0, W, vbias, hbias)
        hneg = self.get_h_from_v(vneg, W, hbias)
        return vneg, hneg

    def _propdown(self, h, W, vbias):
        pre_sigmoid_activation = linear(dropout(h, self.dropout),
                                          W.t(), vbias)
        return sigmoid(pre_sigmoid_activation)

    def _propup(self, v, W, hbias):
        pre_sigmoid_activation = linear(dropout(v, self.dropout), W, hbias)
        return sigmoid(pre_sigmoid_activation)

class PersistentContrastiveDivergence(ContrastiveDivergence):

    def __init__(self, k, n_chains=0, dropout=0, hidden_activations=False,
                 continuous_output=False):
        """Constructor for the class.

        Arguments:

            :param k: The number of iterations in PCD-k
            :type k: int
            :param dropout: Optional parameter, fraction of neurons in the
                            previous layer that are not taken into account when
                            getting a sample.
            :type dropout: float
            :param hidden_activations: Optional parameter to output hidden
                                       activations instead of samples (claimed
                                       to improve learning)
            :type hidden_activations: bool
            :param continuous_output: Optional parameter to output visible
                                      activations instead of samples (for
                                      continuous-valued outputs)
            :type continuous_output: bool
        """
        super().__init__(k, dropout, hidden_activations, continuous_output)
        self.first_call = True
        self.n_chains = n_chains

    def get_v_sample(self, v0, W, vbias, hbias):
        if self.first_call:
            if self.n_chains <= 0:
                self.markov_chains = rand(v0.size()).to(v0.device)
            else:
                self.markov_chains = rand((self.n_chains,) + v0.size()[1:]
                                          ).to(v0.device)
            self.first_call = False
        pre_internal_sampling = self.internal_sampling
        self.internal_sampling = True
        for _ in range(self.k):
            h = self.get_h_from_v(self.markov_chains, W, hbias)
            v = self.get_v_from_h(h, W, vbias)
            self.markov_chains.data = v
        self.internal_sampling = pre_internal_sampling
        return v
