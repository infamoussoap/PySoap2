# PySoap2
Light weight implementation for deep learning.

# Why PySoap2
While Tensorflow and PyTorch is the start of the art software for machine learning, it requires a heavy amount of hardware space to be installed. PySoap2, primarily developed for my own research, is a light weight implementation of fully connected and convolutional neural networks. The only requirements are numpy, abc, inspect, and h5py.

For the GPU version of PySoap2, you will need to install PyOpenCL. This therefore allows PySoap2 to be ran on any GPUs, not just CUDA supported GPUs.

# What is in PySoap2
PySoap2 implements all the layers you will ever need when using neural networks; Dense, Conv_2D, BatchNorm layers, etc. It also features many different optimizers such as Adam, SGD, Adagrad, etc.

# PySoap2 Interface
PySoap is created to feel much like creating a model using Keras. As such, users of Keras will feel at home when using PySoap2

# Whats new in PySoap2 (vs PySoap)
This should just be the same as PySoap but the main change is that we no longer assume the network is a Sequential model. Instead, we now assume that the model can take a tree like structure with the help of Split and Join layers.

Note that Dense, Conv, etc. are not assumed to have 2 children, because I'm not sure how the gradients would work. Instead the tree like structure comes from the fact that split takes 1 parent, and returns 2 childrens. With Join taking 2 parents, returning 1 children

There is now also support for GPU accelerated learning.
