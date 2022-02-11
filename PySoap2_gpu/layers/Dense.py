import numpy as np
from functools import reduce

import pyopencl as cl
import pyopencl.array as cl_array

from PySoap2_gpu.layers import Layer
from PySoap2_gpu.layers import NetworkNode
from PySoap2_gpu.layers.LayerBaseAttributes import LayerBaseAttributes

from .c_code.dense_c_code import dense_source_code
from .ValueChecks import assert_instance_of_cl_array
from .ValueChecks import check_built

from PySoap2_gpu.Exceptions import check_for_valid_context


class DenseInterface:
    """ The interface between the compiled pyopencl-c code with python

        Notes
        -----
        Arguments to all methods are assumed to be stored on the device
    """

    context = None
    queue = None

    program = None

    initialized = False

    def __init__(self, context, queue):
        """ Compile the c-program

            Notes
            -----
            Once this class has been initialized, the c-program will be compiled on the given device context and
            will be bound to the class (not instances of the class).
            It will no longer be possible to re-initialize this class again.
        """
        if DenseInterface.initialized:
            return

        DenseInterface.context = context
        DenseInterface.queue = queue

        DenseInterface.program = cl.Program(context, dense_source_code).build()

        DenseInterface.initialized = True

    @staticmethod
    def predict(z, W, b, input_length, output_length, out):
        check_for_valid_context(DenseInterface.context, z, W, b, out)

        device_global_shape = out.shape
        event = DenseInterface.program.predict(DenseInterface.queue, device_global_shape,
                                               None,
                                               z.data, W.data, b.data, input_length,
                                               output_length, out.data)
        event.wait()

    @staticmethod
    def delta_back_prop(g_prime, new_delta, W, input_length, output_length, out):
        check_for_valid_context(DenseInterface.context, g_prime, new_delta, W, out)

        device_global_shape = g_prime.shape
        event = DenseInterface.program.delta_back_prop(DenseInterface.queue,
                                                       device_global_shape, None,
                                                       g_prime.data, new_delta.data, W.data,
                                                       input_length, output_length, out.data)
        event.wait()

    @staticmethod
    def weight_gradient(delta, prev_z, input_length, output_length, N, out):
        check_for_valid_context(DenseInterface.context, delta, prev_z, out)

        device_global_shape = (output_length, input_length)  # Same shape as the weight matrix
        event = DenseInterface.program.weight_gradient(DenseInterface.queue,
                                                       device_global_shape, None,
                                                       delta.data, prev_z.data, input_length,
                                                       output_length, N, out.data)
        event.wait()

    @staticmethod
    def bias_gradient(delta, output_length, N, out):
        check_for_valid_context(DenseInterface.context, delta, out)

        device_global_shape = (output_length,)
        event = DenseInterface.program.bias_gradient(DenseInterface.queue,
                                                     device_global_shape, None, delta.data,
                                                     output_length, N, out.data)
        event.wait()


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

        n = len(z_device)

        out_device = cl_array.empty(self.queue, (n, *self.output_shape), dtype=np.float64)

        DenseInterface.predict(z_device, self.W, self.b, self.input_length_device,
                               self.output_length_device, out_device)

        if output_only:
            return self.activation_function_(out_device)
        return out_device, self.activation_function_(out_device)

    @check_built
    def get_delta_backprop_(self, g_prime_device, new_delta, *args):
        assert_instance_of_cl_array(g_prime_device)

        out_device = cl_array.empty(self.queue, g_prime_device.shape, dtype=np.float64)

        summed_delta_device = reduce(lambda x, y: x + y, new_delta)

        DenseInterface.delta_back_prop(g_prime_device, summed_delta_device, self.W,
                                       self.input_length_device, self.output_length_device, out_device)

        return out_device

    @check_built
    def get_parameter_gradients_(self, delta_device, z_device, e=1e-7):
        assert_instance_of_cl_array(z_device)

        summed_delta_device = reduce(lambda x, y: x + y, delta_device)

        N = np.int32(len(z_device))

        W_grad_device = cl_array.empty(self.queue, self.W.shape, dtype=np.float64)
        DenseInterface.weight_gradient(summed_delta_device, z_device, self.input_length_device,
                                       self.output_length_device, N, W_grad_device)

        b_grad_device = cl_array.empty(self.queue, self.b.shape, dtype=np.float64)
        DenseInterface.bias_gradient(summed_delta_device, self.output_length_device, N, b_grad_device)

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
