import numpy as np
from pyopencl import clmath

from PySoap2_gpu.utils import ClArrayTricks


def mse(predictions, targets, grad=False):
    N = predictions.shape[0]
    if grad:
        return (predictions - targets) / N
    return np.sum(((predictions - targets) ** 2) / (2 * N))


def cross_entropy(predictions, targets, epsilon=1e-12, grad=False):
    ClArrayTricks.clip_cl_array_in_place(predictions, epsilon, 1.0 - epsilon)
    if grad:
        return -targets / predictions + (1.0 - targets) / (1.0 - predictions)

    N = predictions.shape[0]
    ce = -np.sum(targets * clmath.log(predictions + 1e-9)) / N
    return ce


error_functions = {'mse': mse,
                   'cross_entropy': cross_entropy}
