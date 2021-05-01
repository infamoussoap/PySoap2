import numpy as np

import pyopencl as cl
import pyopencl.array as cl_array
from pyopencl import clmath

from functools import partial

from PySoap2_gpu.utils import clip_cl_array_in_place

from PySoap2_gpu.utils import cl_math_functions


def get_activation_function(device_context, device_queue, name):
    if device_context is None:
        raise ValueError('Context for gpu cannot be None.')

    if name == 'relu':
        def relu(x_device, grad=False):
            """ x is assumed to be an instance of cl.array.Array"""
            out_device = cl_array.empty_like(x_device)
            if grad:
                cl_math_functions.elementwise_relu_grad(device_context)(x_device, out_device)
            else:
                cl_math_functions.elementwise_relu(device_context)(x_device, out_device)
            return out_device

        return relu

    elif name == 'linear':
        def linear(x_device, grad=False):
            if grad:
                return cl_array.zeros_like(x_device) + 1.0
            return x_device
        return linear
    elif name == 'softmax':
        return partial(cl_math_functions.softmax_gpu, device_context, device_queue)

    elif name == 'sigmoid':
        def sigmoid(x_device, grad=False):
            out_device = cl_array.empty_like(x_device)
            cl_math_functions.elementwise_sigmoid(device_context)(x_device, out_device)
            if grad:
                return out_device * (1 - out_device)
            return out_device

        return sigmoid

    else:
        raise Exception(f'{name} is not a defined function.')


def get_error_function(device_context, device_queue, name):
    if name == 'mse':
        def mse(predictions, targets, grad=False):
            if grad:
                return 2 * (predictions - targets)
            N = predictions.shape[0]
            return np.sum(((predictions - targets) ** 2) / 2) / N

        return mse
    elif name == 'cross_entropy':
        def cross_entropy(predictions, targets, epsilon=1e-12, grad=False):
            clip_cl_array_in_place(device_context, predictions, epsilon, 1.0 - epsilon)

            if grad:
                return -targets / predictions + (1.0 - targets) / (1.0 - predictions)

            N = predictions.shape[0]
            ce = -np.sum(targets * clmath.log(predictions + 1e-9)) / N
            return ce

        return cross_entropy


def get_metric_function(device_context, device_queue, name):
    if name == 'accuracy':
        def accuracy(predictions, target):
            predicted_labels = cl_math_functions.arg_max_across_last_axis(device_context, device_queue, predictions)
            target_labels = cl_math_functions.arg_max_across_last_axis(device_context, device_queue, target)

            N = len(predictions)
            return cl_array.sum((predicted_labels == target_labels)).astype(np.float32)/N

        return accuracy
    else:
        raise Exception(f'{name} is not a defined metric.')
