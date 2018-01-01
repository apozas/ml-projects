# Energy-based models in Pytorch

Author: Alejandro Pozas Kerstjens

Implementation of different generative models based on energy-based learning. Examples are provided
with the MNIST dataset.

Libraries required:
- [pytorch](http://www.pytorch.org) as DL framework
- [numpy](http://www.numpy.org/) for math operations
- [matplotlib](https://matplotlib.org/) for image visualization
- [tqdm](https://pypi.python.org/pypi/tqdm) for custom progress bar
- [imageio](http://imageio.github.io/) for exporting outputs to ``.gif`` format

1. RBM
[Restricted Boltzmann Machine](http://stanford.edu/~jlmcc/papers/PDP/Volume%201/Chap6_PDP86.pdf)
with binary visible and hidden units. Although in this example it is used as a generative model,
RBMs can also perform supervised tasks.
### Example
![RBM](./rbm_sample.gif)

2. Continuous RBM
Restricted Boltzmann Machine with binary hidden but continuous visible units.
### Example
![CRBM](./crbm_sample.gif)

3. To-Do list
- [ ] [Deep Belief Network](http://www.scholarpedia.org/article/Deep_belief_networks) with binary
visible units
- [ ] Deep Belief Network with continuous visible units 