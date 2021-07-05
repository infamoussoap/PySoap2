import numpy as np
from functools import reduce

import pyopencl as cl
import pyopencl.array as cl_array

from PySoap2_gpu.layers import Layer
from PySoap2_gpu.layers import NetworkNode
from PySoap2_gpu.layers.LayerBaseAttributes import LayerBaseAttributes

from .c_code.dropout_c_code import dropout_source_code
from .ValueChecks import assert_instance_of_cl_array
from .ValueChecks import check_built


class DropoutInterfaceToDevice:
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
        if DropoutInterfaceToDevice.initialized:
            return

        DropoutInterfaceToDevice.device_context = device_context
        DropoutInterfaceToDevice.device_queue = device_queue

        DropoutInterfaceToDevice.device_program = cl.Program(device_context, dropout_source_code).build()

        DropoutInterfaceToDevice.initialized = True

    @staticmethod
    def dropout(z, mask, output_length, out):
        device_global_shape = (len(z), output_length)

        event = DropoutInterfaceToDevice.device_program.dropout(DropoutInterfaceToDevice.device_queue,
                                                                device_global_shape, None,
                                                                z.data, mask.data, output_length, out.data)
        event.wait()


class Dropout(NetworkNode, LayerBaseAttributes, Layer):
    def __init__(self, rate):
        """ rate : Float between 0 and 1. The fraction of inputs to drop """
        NetworkNode.__init__(self)
        LayerBaseAttributes.__init__(self)

        self.rate = rate
        self.mask = None

    def build(self, device_context, device_queue):
        self.device_context = device_context
        self.device_queue = device_queue

        if not DropoutInterfaceToDevice.initialized:
            DropoutInterfaceToDevice(device_context, device_queue)

        parent = self.parents[0]
        self.input_shape = parent.output_shape
        self.output_shape = parent.output_shape

        self.built = True

    @check_built
    def predict(self, z, output_only=True, pre_activation_of_input=None, training=False):
        if not training:
            if output_only:
                return z
            return pre_activation_of_input, z

        self.mask = cl_array.to_device(self.device_queue, np.random.rand(*self.input_shape) > self.rate)

        # The inverted dropout method, where scaling is performed during training, so the
        # forward pass, during testing, does not need to be scaled.
        # see https://cs231n.github.io/neural-networks-2/

        z_inverted_dropout = self._dropout(z, self.mask, self.output_length_device) / (1 - self.rate)
        if output_only:
            return z_inverted_dropout
        return z_inverted_dropout, z_inverted_dropout

    @staticmethod
    def _dropout(z, mask, output_length):
        """ z, mask assumed to be cl_arrays """
        out_device = cl_array.empty_like(z)
        DropoutInterfaceToDevice.dropout(z, mask, output_length, out_device)
        return out_device

    @check_built
    def get_delta_backprop_(self, g_prime, new_delta, prev_z):
        delta = reduce(lambda x, y: x + y, new_delta)
        return self._dropout(delta, self.mask, self.output_length_device)

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
        return f'Dropout', f'Output Shape {(None, *self.output_shape)}'
