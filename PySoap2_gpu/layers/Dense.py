import numpy as np
from functools import reduce

import pyopencl.array as cl_array

from PySoap2_gpu.layers import Layer
from PySoap2_gpu.layers import NetworkNode
from PySoap2_gpu.layers.LayerBaseAttributes import LayerBaseAttributes

from .ValueChecks import assert_instance_of_cl_array
from .ValueChecks import check_built

from PySoap2_gpu.Exceptions import check_for_valid_context

from PySoap2_gpu.layers.ProgramInterface.DenseInterface import DenseInterface


class Dense(NetworkNode, LayerBaseAttributes, Layer):
    """ A Dense layer that only performs computations on the GPU (or device)

        For more details, check documentation in PySoap2
    """

    def __init__(self, hidden_nodes, activation_function, *arg, activation_kwargs=None, weight_decay=0.0, **kwargs):
        """ A fully connected layer """
        NetworkNode.__init__(self)
        LayerBaseAttributes.__init__(self)

        self.hidden_nodes = hidden_nodes

        self.activation_function = activation_function
        self.activation_kwargs = {} if activation_kwargs is None else activation_kwargs

        self.weight_decay = weight_decay

        self.W = None
        self.b = None

    def build(self, device_context, device_queue):
        """ Initialises the weight and bias units """

        self.context = device_context
        self.queue = device_queue

        DenseInterface(self.context, self.queue)

        input_shape = self.parents[0].output_shape

        self.output_shape = (self.hidden_nodes,)
        self.input_shape = input_shape

        if len(self.input_shape) > 1:
            raise ValueError('Input shape to Dense layer must be 1-dimensional, but is '
                             f'{len(self.input_shape)}-dimensional.')

        # Initialise the the weight with Glorot-Uniform, a uniform distribution over [-limit, limit],
        # where limit = sqrt(6 / (fan_in + fan_out)) (fan_in is the number of input units in the weight
        # tensor and fan_out is the number of output units).
        limit = np.sqrt(6 / (np.prod(self.input_shape) + np.prod(self.output_shape)))
        W = np.random.uniform(low=-limit, high=limit, size=(*self.output_shape, *input_shape)).astype(np.float64)
        b = np.zeros(self.output_shape).astype(np.float64)

        self.W = cl_array.to_device(self.queue, W)
        self.b = cl_array.to_device(self.queue, b)

        self.built = True

    @check_built
    def predict(self, z_device, output_only=True, **kwargs):
        assert_instance_of_cl_array(z_device)

        out = DenseInterface.predict(z_device, self.W, self.b)

        if output_only:
            return self.activation_function_(out)
        return out, self.activation_function_(out)

    @check_built
    def get_delta_backprop_(self, g_prime_device, new_delta, *args):
        assert_instance_of_cl_array(g_prime_device)

        delta = reduce(lambda x, y: x + y, new_delta)

        return DenseInterface.delta_back_prop(g_prime_device, delta, self.W)

    @check_built
    def get_parameter_gradients_(self, delta_device, z_device, e=1e-7):
        assert_instance_of_cl_array(z_device)

        delta = reduce(lambda x, y: x + y, delta_device)

        W_grad_device = DenseInterface.weight_gradient(delta, z_device)
        b_grad_device = DenseInterface.bias_gradient(delta)

        if abs(self.weight_decay) > e:
            parameter_gradients = {'weight': W_grad_device + self.weight_decay * self.W,
                                   'bias': b_grad_device}
        else:
            parameter_gradients = {'weight': W_grad_device, 'bias': b_grad_device}
        return parameter_gradients

    @check_built
    def update_parameters_(self, parameter_updates):
        self.W -= parameter_updates['weight']
        self.b -= parameter_updates['bias']

    @check_built
    def get_weights(self, as_dict=False):
        if as_dict:
            return {'W': self.W.get(), 'b': self.b.get()}
        return self.W.get(), self.b.get()

    @check_built
    def summary_(self):
        return f'Dense {(self.hidden_nodes,)}', f'Output Shape {(None, *self.output_shape)}'

    def __str__(self):
        return f'Dense: Output Shape {(None, *self.output_shape)}'
