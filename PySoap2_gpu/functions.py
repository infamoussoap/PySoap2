import numpy as np

import pyopencl as cl
import pyopencl.array as cl_array
from pyopencl import clmath

from functools import partial

from PySoap2_gpu.utils import ClMathFunctions
from PySoap2_gpu.utils import ClArrayTricks


class ActivationFunction:
    initialized = False

    device_context = None
    device_queue = None

    def __init__(self, device_context, device_queue):
        # If this class is initialized, it means that the programs is already on the device
        if ActivationFunction.initialized:
            return

        ActivationFunction.device_context = device_context
        ActivationFunction.device_queue = device_queue

        ActivationFunction.initialized = True

        ClMathFunctions(device_context, device_queue)
        ClArrayTricks(device_context, device_queue)

    @staticmethod
    def get_activation_function(name):
        if name == 'relu':
            def relu(x_device, grad=False):
                if grad:
                    return ClMathFunctions.relu_grad(x_device)
                return ClMathFunctions.relu(x_device)

            return relu
        elif name == 'linear':
            def linear(x_device, grad=False):
                if grad:
                    return cl_array.zeros_like(x_device) + 1.0
                return x_device

            return linear
        elif name == 'softmax':
            def softmax(x_device, grad=False):
                softmax_val = ClMathFunctions.softmax(x_device)
                if grad:
                    return softmax_val * (1 - softmax_val)
                return softmax_val

            return softmax
        elif name == 'sigmoid':
            def sigmoid(x_device, grad=False):
                sigmoid_val = ClMathFunctions.sigmoid(x_device)
                if grad:
                    return sigmoid_val * (1 - sigmoid_val)
                return sigmoid_val

            return sigmoid

        else:
            raise Exception(f'{name} is not a defined function.')


class ErrorFunction:
    initialized = False

    device_context = None
    device_queue = None

    def __init__(self, device_context, device_queue):
        ErrorFunction.device_queue = device_queue
        ErrorFunction.device_context = device_context

        ClMathFunctions(device_context, device_queue)
        ClArrayTricks(device_context, device_queue)

        ErrorFunction.initialized = True

    @staticmethod
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
                ClArrayTricks.clip_cl_array_in_place(predictions, epsilon, 1.0 - epsilon)
                if grad:
                    return -targets / predictions + (1.0 - targets) / (1.0 - predictions)

                N = predictions.shape[0]
                ce = -np.sum(targets * clmath.log(predictions + 1e-9)) / N
                return ce

            return cross_entropy


class MetricFunction:
    initialized = False

    device_context = None
    device_queue = None

    def __init__(self, device_context, device_queue):
        MetricFunction.device_queue = device_queue
        MetricFunction.device_context = device_context

        if not ClArrayTricks.initialized:
            ClArrayTricks(device_context, device_queue)

        MetricFunction.initialized = True

    @staticmethod
    def get_metric_function(name):
        if name == 'accuracy':
            def accuracy(predictions, target):
                predicted_labels = ClArrayTricks.arg_max_across_last_axis(predictions)
                target_labels = ClArrayTricks.arg_max_across_last_axis(target)

                # N = len(predictions)
                # return cl_array.sum((predicted_labels == target_labels)).astype(np.float32)/N
                return np.mean((predicted_labels == target_labels).get())

            return accuracy
        else:
            raise Exception(f'{name} is not a defined metric.')
