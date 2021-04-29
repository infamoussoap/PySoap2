import numpy as np

import pyopencl as cl
import pyopencl.array as cl_array

from PySoap2_gpu.layers import Layer
from PySoap2_gpu.layers.NetworkNode import NetworkNode
from PySoap2_gpu.layers.LayerBaseAttributes import LayerBaseAttributes

from .dense_c_code import dense_source_code
from .ValueChecks import assert_instance_of_cl_array


class Dense(NetworkNode, LayerBaseAttributes, Layer):
    """ A fully connected layer

        Parameters
        ----------
        hidden_nodes : int
            The number of neurons in this layer
        activation_function : str
            The name of the activation function of this layer
        activation_kwargs : dict of str - :obj:, optional
            The keyword arguments for the activation function if it has hyper-parameters
    """

    def __init__(self, hidden_nodes, activation_function, *arg, activation_kwargs=None, **kwargs):
        NetworkNode.__init__(self)
        LayerBaseAttributes.__init__(self)

        self.hidden_nodes = hidden_nodes

        self.activation_function = activation_function
        self.activation_kwargs = {} if activation_kwargs is None else activation_kwargs

        self.W_device = None
        self.b_device = None

        self.input_length_device = None
        self.output_length_device = None

    def build(self, gpu_context, gpu_queue):
        """ Initialises the weight and bias units """

        self.gpu_context = gpu_context
        self.gpu_queue = gpu_queue

        input_shape = self.parents[0].output_shape

        self.output_shape = (self.hidden_nodes,)
        self.input_shape = input_shape

        self.input_length_device = cl_array.to_device(self.gpu_queue, np.array(self.input_shape[0], dtype=np.int32))
        self.output_length_device = cl_array.to_device(self.gpu_queue, np.array(self.output_shape[0], dtype=np.int32))

        # Initialise the the weight with Glorot-Uniform, a uniform distribution over [-limit, limit],
        # where limit = sqrt(6 / (fan_in + fan_out)) (fan_in is the number of input units in the weight
        # tensor and fan_out is the number of output units).
        limit = np.sqrt(6 / (np.prod(self.input_shape) + np.prod(self.output_shape)))
        W = np.random.uniform(low=-limit, high=limit, size=(*self.output_shape, *input_shape)).astype(np.float32)
        b = np.zeros(self.output_shape).astype(np.float32)

        self.W_device = cl_array.to_device(self.gpu_queue, W)
        self.b_device = cl_array.to_device(self.gpu_queue, b)

        self.gpu_program = cl.Program(self.gpu_context, dense_source_code).build()

        self.built = True

    def predict(self, z_device, output_only=True, **kwargs):
        assert_instance_of_cl_array(z_device)

        N = len(z_device)

        out_device = cl_array.empty(self.gpu_queue, (N, *self.output_shape), dtype=np.float32)

        args = [z_device, self.W_device, self.b_device, self.input_length_device, self.output_length_device,
                out_device]
        args_data = [arg.data for arg in args]

        self.gpu_program.predict(self.gpu_queue, (N, *self.output_shape), None, *args_data)

        if output_only:
            return self.activation_function_(out_device)
        return out_device, self.activation_function_(out_device)

    def get_delta_backprop_(self, g_prime_device, delta_device, *args):
        assert_instance_of_cl_array(g_prime_device)
        assert_instance_of_cl_array(delta_device)

        out_device = cl_array.empty(self.gpu_queue, g_prime_device.shape, dtype=np.float32)

        args = [g_prime_device, delta_device, self.W_device, self.input_length_device,
                self.output_length_device, out_device]
        args_data = [arg.data for arg in args]

        self.gpu_program.delta_back_prop(self.gpu_queue, g_prime_device.shape, None, *args_data)

        return out_device

    def get_parameter_gradients_(self, delta_device, z_device):
        assert_instance_of_cl_array(delta_device)
        assert_instance_of_cl_array(z_device)

        N = np.array(len(z_device)).astype(np.int32)
        N_device = cl_array.to_device(self.gpu_queue, N)

        W_grad_device = cl_array.empty(self.gpu_queue, self.W_device.shape, dtype=np.float32)
        args = [delta_device, z_device, self.input_length_device, self.output_length_device,
                N_device, W_grad_device]
        args_data = [arg.data for arg in args]
        self.gpu_program.weight_gradient(self.gpu_queue, self.W_device.shape, None, *args_data)

        b_grad_device = cl_array.empty(self.gpu_queue, self.b_device.shape, dtype=np.float32)
        args = [delta_device, self.output_length_device, N_device, b_grad_device]
        args_data = [arg.data for arg in args]
        self.gpu_program.bias_gradient(self.gpu_queue, self.output_shape, None, *args_data)

        parameter_gradients = {'weight': W_grad_device, 'bias': b_grad_device}
        return parameter_gradients

    def update_parameters_(self, parameter_updates):
        raise NotImplementedError

    def get_weights(self):
        raise NotImplementedError

    def summary_(self):
        return f'Dense {(self.hidden_nodes,)}', f'Output Shape {(None, *self.output_shape)}'

    def __str__(self):
        return f'Dense: Output Shape {(None, *self.output_shape)}'
