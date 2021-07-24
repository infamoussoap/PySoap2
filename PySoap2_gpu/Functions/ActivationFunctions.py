import pyopencl.array as cl_array

from PySoap2_gpu.utils import ClMathFunctions


def relu(x_device, grad=False):
    if grad:
        return ClMathFunctions.relu_grad(x_device)
    return ClMathFunctions.relu(x_device)


def linear(x_device, grad=False):
    if grad:
        return cl_array.zeros_like(x_device) + 1.0
    return x_device


def softmax(x_device, grad=False):
    softmax_val = ClMathFunctions.softmax(x_device)
    if grad:
        return softmax_val * (1 - softmax_val)
    return softmax_val


def sigmoid(x_device, grad=False):
    sigmoid_val = ClMathFunctions.sigmoid(x_device)
    if grad:
        return sigmoid_val * (1 - sigmoid_val)
    return sigmoid_val


def log_softmax(x_device, grad=False):
    if grad:
        return 1 - ClMathFunctions.softmax(x_device)
    return ClMathFunctions.log_softmax(x_device)


activation_functions = {'relu': relu,
                        'softmax': softmax,
                        'linear': linear,
                        'sigmoid': sigmoid}
