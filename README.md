# Tensor networks for unsupervised machine learning

Implementations of [Tensor networks for unsupervised machine learning](https://arxiv.org/abs/2106.12974) in PyTorch.

The code is tested on RTX 2080Ti and Tesla V100 with `Python 3.6.8`, `PyTorch 1.4.0` and `CUDA 10.1`.

The `src_amps` contains code for AMPS, the `src_conv_amps` contains code for Convolution AMPS, the `src_rbm` contains code for RMB and the `src_rwd` contains code for real world datasets experiments. To run the MNIST example, please download [binarized mnist dataset](https://github.com/mgermain/MADE/releases/download/ICML2015/binarized_mnist.npz) and put it in `./data`. 

The implementations of MPS and LPS can be found in [tensor_networks_for_probabilistic_modeling](https://github.com/glivan/tensor_networks_for_probabilistic_modeling). The implementations of VAN can be found in [stat-mech-van](https://github.com/wdphy16/stat-mech-van).