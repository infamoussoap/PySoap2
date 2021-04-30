import pyopencl as cl
import pyopencl.array as cl_array
from pyopencl import clmath

import numpy as np


def get_activation_function(name, gpu_context):
    if gpu_context is None:
        raise ValueError('Context for gpu cannot be None.')

    if name == 'relu':
        elementwise_relu = cl.elementwise.ElementwiseKernel(gpu_context, "float *x, float *out",
                                                            "out[i] = x[i] > 0 ? x[i] : 0.0", "relu")
        elementwise_relu_grad = cl.elementwise.ElementwiseKernel(gpu_context, "float *x, float *out",
                                                                 "out[i] = x[i] > 0 ? 1.0 : 0.0", "relu")

        def relu(x_device, grad=False):
            """ x is assumed to be an instance of cl.array.Array"""
            out_device = cl_array.empty_like(x_device)
            if grad:
                elementwise_relu_grad(x_device, out_device)
            else:
                elementwise_relu(x_device, out_device)
            return out_device

        return relu

    elif name == 'linear':
        def linear(x_device, grad=False):
            if grad:
                return cl_array.zeros_like(x_device) + 1.0
            return x_device

    elif name == 'sigmoid':
        elementwise_sigmoid = cl.elementwise.ElementwiseKernel(gpu_context,
                                                               "float *x, float *out",
                                                               """
                                                               out[i] = SIGMOID(x[i])
                                                               """,
                                                               "sigmoid",
                                                               preamble='#define SIGMOID(x) x > 0 ? '
                                                                        '1.0/(1.0 + exp(-x)) : exp(x) / (exp(x) + 1.0))'
                                                               )

        def sigmoid(x_device, grad=False):
            out_device = cl_array.empty_like(x_device)
            elementwise_sigmoid(x_device, out_device)
            if grad:
                return out_device * (1 - out_device)
            return out_device

        return sigmoid

    else:
        raise Exception(f'{name} is not a defined function.')


def get_error_function(name):
    if name == 'mse':
        def mse(predictions, targets, grad=False):
            if grad:
                return 2 * (predictions - targets)
            N = predictions.shape[0]
            return np.sum(((predictions - targets) ** 2) / 2) / N

        return mse
    elif name == 'cross_entropy':
        def cross_entropy(predictions, targets, epsilon=1e-12, grad=False):
            predictions = np.clip(predictions, epsilon, 1. - epsilon)  # Need to fix this line

            if grad:
                return -targets / predictions + (1 - targets) / (1 - predictions)

            N = predictions.shape[0]
            ce = -np.sum(targets * clmath.log(predictions + 1e-9)) / N
            return ce

        return cross_entropy


def get_metric_function(name):
    pass
