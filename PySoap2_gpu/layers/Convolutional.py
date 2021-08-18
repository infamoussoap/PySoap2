import numpy as np
from functools import reduce

import pyopencl as cl
import pyopencl.array as cl_array

from PySoap2_gpu.layers import Layer
from PySoap2_gpu.layers import NetworkNode
from PySoap2_gpu.layers.LayerBaseAttributes import LayerBaseAttributes

from .c_code.conv2d_c_code import conv2d_source_code
from .ValueChecks import assert_instance_of_cl_array
from .ValueChecks import check_built

from PySoap2_gpu.Exceptions import check_for_valid_context


class Conv2DInterfaceToDevice:
    device_context = None
    device_queue = None

    device_program = None

    initialized = False

    def __init__(self, device_context, device_queue):
        """ Compile the c-program

            Notes
            -----
            Once this class has been initialized, the c-program will be compiled on the given device context and
            will be bound to the class (not instances of the class).
            It will no longer be possible to re-initialize this class again.
        """
        if Conv2DInterfaceToDevice.initialized:
            return

        Conv2DInterfaceToDevice.device_context = device_context
        Conv2DInterfaceToDevice.device_queue = device_queue

        Conv2DInterfaceToDevice.device_program = cl.Program(device_context, conv2d_source_code).build()

        Conv2DInterfaceToDevice.initialized = True

    @staticmethod
    def predict(z, filter_, bias_,
                filter_height, filter_width, num_of_filters,
                stride,
                image_width, image_depth,
                output_width,
                input_length, output_length,
                out):

        output_height = int(output_length / (output_width * num_of_filters))

        program = Conv2DInterfaceToDevice.device_program
        queue = Conv2DInterfaceToDevice.device_queue

        for i in range(num_of_filters):
            current_filter = np.int32(i)

            event = program.conv2d(queue, (len(z), output_height, output_width), None,
                                   z.data, filter_.data, bias_.data,
                                   filter_height, filter_width, num_of_filters,
                                   stride, current_filter,
                                   image_width, image_depth,
                                   output_width,
                                   input_length, output_length,
                                   out.data)
        queue.finish()

    @staticmethod
    def delta_back_prop(delta, eye_conv, g_prime, input_length, output_length, out):
        program = Conv2DInterfaceToDevice.device_program
        queue = Conv2DInterfaceToDevice.device_queue

        global_shape = (np.prod(out.shape),)

        event = program.delta_back_prop(queue, global_shape, None,
                                        delta.data, eye_conv.data, g_prime.data, input_length, output_length, out.data)

        event.wait()
        return out

    @staticmethod
    def filter_gradient(prev_z, delta,
                        output_height, output_width, num_of_filters,
                        stride,
                        image_width, image_depth,
                        N, input_length,
                        filter_width,
                        out):
        program = Conv2DInterfaceToDevice.device_program
        queue = Conv2DInterfaceToDevice.device_queue

        global_shape = (np.prod(out.shape),)

        event = program.filter_gradient(queue, global_shape, None,
                                        prev_z.shape, delta.shape,
                                        output_height, output_width, num_of_filters,
                                        stride,
                                        image_width, image_depth,
                                        N, input_length,
                                        filter_width,
                                        out.data)
        event.wait()

    @staticmethod
    def bias_gradient(delta, sum_length, num_of_filters, out):
        program = Conv2DInterfaceToDevice.device_program
        queue = Conv2DInterfaceToDevice.device_queue

        global_shape = out.shape

        event = program.bias_gradient(queue, global_shape, None,
                                      delta.data, sum_length, num_of_filters, out.data)
        event.wait()


class Conv2D(NetworkNode, LayerBaseAttributes, Layer):
    def __init__(self, filter_num, filter_spatial_shape, stride, activation_function, activation_kwargs=None):
        NetworkNode.__init__(self)
        LayerBaseAttributes.__init__(self)

        self.filter_num = filter_num
        self.filter_spatial_shape = filter_spatial_shape

        self.stride = stride

        self.activation_function = activation_function
        self.activation_kwargs = {} if activation_kwargs is None else activation_kwargs

        self.single_filter_shape = None
        self.filter_shape = None
        self.output_spatial_shape = None

        self.filter = None
        self.b = None

    def build(self, device_context, device_queue):
        raise NotImplementedError

    @check_built
    def predict(self, z, output_only=True, **kwargs):
        raise NotImplementedError

    @check_built
    def get_delta_backprop_(self, g_prime, new_delta, prev_z):
        raise NotImplementedError

    @check_built
    def get_parameter_gradients_(self, delta, prev_z):
        raise NotImplementedError

    @check_built
    def update_parameters_(self, parameter_updates):
        raise NotImplementedError

    @check_built
    def get_weights(self, as_dict=False):
        if as_dict:
            return {'filter': self.filter.get(), 'b': self.b.get()}
        return self.filter.get(), self.b.get()

    @check_built
    def summary_(self):
        return f'Conv2D {self.filter_num} x {(self.single_filter_shape,)}', f'Output Shape {(None, *self.output_shape)}'
