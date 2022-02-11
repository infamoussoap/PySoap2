import numpy as np
from functools import reduce

import warnings

import pyopencl.array as cl_array

from PySoap2_gpu.layers import Layer
from PySoap2_gpu.layers import NetworkNode
from PySoap2_gpu.layers.LayerBaseAttributes import LayerBaseAttributes

from .ValueChecks import check_built

from PySoap2_gpu.layers.ProgramInterface.MaxPoolingInterface import MaxPoolingInterface


class MaxPooling2D(NetworkNode, LayerBaseAttributes, Layer):
    def __init__(self, window_shape, stride):
        NetworkNode.__init__(self)
        LayerBaseAttributes.__init__(self)

        if window_shape[0] > stride or window_shape[1] > stride:
            warnings.warn("Backpropagation of this layer is not gauranteed to work when the windows overlap "
                          f"as atomic add is not implemented yet. To gaurantee correctness, make the window shape "
                          f"dimensions: {window_shape} is >= to the stride: {stride}.")

        self.window_shape = window_shape
        self.stride = stride

        self.max_indices = None

    def build(self, device_context, device_queue):
        self.context = device_context
        self.queue = device_queue

        if not MaxPoolingInterface.initialized:
            MaxPoolingInterface(device_context, device_queue)

        parent = self.parents[0]
        self.input_shape = parent.output_shape

        n = np.int32((self.input_shape[0] - self.window_shape[0]) / self.stride + 1)
        m = np.int32((self.input_shape[1] - self.window_shape[1]) / self.stride + 1)
        self.output_shape = (n, m, self.input_shape[2])

        self.built = True

    @check_built
    def predict(self, z, output_only=True, pre_activation_of_input=None, training=False):
        max_val, max_arg = MaxPoolingInterface.maxpool_2d(z, self.window_shape, self.stride)

        if training:
            self.max_indices = max_arg

        if output_only:
            return max_val
        return max_val, max_val

    @check_built
    def get_delta_backprop_(self, g_prime, new_delta, prev_z):
        delta = reduce(lambda x, y: x + y, new_delta)

        N = len(g_prime)
        dx = cl_array.zeros(self.queue, (N, *self.input_shape), np.float64)
        MaxPoolingInterface.add_at(dx, self.max_indices, delta)

        return dx

    @check_built
    def get_parameter_gradients_(self, delta, prev_z):
        return {}

    @check_built
    def update_parameters_(self, parameter_updates):
        pass

    @check_built
    def get_weights(self, as_dict=False):
        if as_dict:
            return {}
        return None

    @check_built
    def summary_(self):
        return f"MaxPooling2D", f"Output Shape {self.output_shape}"
