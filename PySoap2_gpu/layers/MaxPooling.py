import numpy as np
from functools import reduce

import warnings

import pyopencl as cl
import pyopencl.array as cl_array

from PySoap2_gpu.layers import Layer
from PySoap2_gpu.layers import NetworkNode
from PySoap2_gpu.layers.LayerBaseAttributes import LayerBaseAttributes

from .c_code.maxpooling_c_code import maxpool_source_code
from .ValueChecks import check_built

from PySoap2_gpu.Exceptions import check_for_valid_context


class MaxPoolingInterface:
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
        if MaxPoolingInterface.initialized:
            return

        MaxPoolingInterface.context = context
        MaxPoolingInterface.queue = queue

        MaxPoolingInterface.program = cl.Program(context, maxpool_source_code).build()

        MaxPoolingInterface.initialized = True

    @staticmethod
    def maxpool_2d(z, window_shape, stride):
        queue = MaxPoolingInterface.queue
        program = MaxPoolingInterface.program

        image_shape = z.shape[1:]
        n = np.int32((image_shape[0] - window_shape[0]) / stride + 1)
        m = np.int32((image_shape[1] - window_shape[1]) / stride + 1)
        output_shape = (n, m, image_shape[2])

        max_out = cl_array.zeros(queue, (len(z), *output_shape), np.float64)
        argmax_out = cl_array.zeros(queue, (len(z), *output_shape), np.int32)

        image_shape = [np.int32(x) for x in image_shape]
        output_shape = [np.int32(x) for x in output_shape]
        window_shape = [np.int32(x) for x in window_shape]

        image_length = np.int32(np.prod(image_shape))
        output_length = np.int32(np.prod(output_shape))

        events = []
        global_shape = (len(z), n, m)
        for current_channel in range(image_shape[2]):
            event = program.maxpool_2d(queue, global_shape, None,
                                       z.data,
                                       *image_shape,
                                       *window_shape, np.int32(stride),
                                       np.int32(current_channel),
                                       n, m,
                                       image_length, output_length,
                                       max_out.data, argmax_out.data)
            events.append(event)

        cl.wait_for_events(events)
        return max_out, argmax_out

    @staticmethod
    def add_at(z, index, vals):
        queue = MaxPoolingInterface.queue
        program = MaxPoolingInterface.program

        global_shape = (int(np.prod(vals.shape)),)
        event = program.add_at(queue, global_shape, None,
                               z.data, index.data, vals.data)
        event.wait()
        return z


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
        self.device_context = device_context
        self.device_queue = device_queue

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
        dx = cl_array.zeros(self.device_queue, (N, *self.input_shape), np.float64)
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
