import numpy as np
from functools import reduce

import pyopencl as cl
import pyopencl.array as cl_array

from PySoap2_gpu.layers import Layer
from PySoap2_gpu.layers import NetworkNode
from PySoap2_gpu.layers.LayerBaseAttributes import LayerBaseAttributes

from .c_code.split_c_code import split_source_code

from .ValueChecks import check_built


class SplitInterfaceToDevice:
    device_context = None
    device_queue = None
    device_program = None

    initialized = False

    def __init__(self, device_context, device_queue):
        if SplitInterfaceToDevice.initialized:
            return

        SplitInterfaceToDevice.device_context = device_context
        SplitInterfaceToDevice.device_queue = device_queue

        SplitInterfaceToDevice.device_program = cl.Program(device_context, split_source_code).build()

        SplitInterfaceToDevice.initialized = True

    @staticmethod
    def get_input_at_mask(input_, mask_positions, input_length, output_length, output_):
        device_global_shape = output_.shape

        event = SplitInterfaceToDevice.device_program.get_input_at_mask(SplitInterfaceToDevice.device_queue,
                                                                        device_global_shape, None,
                                                                        input_.data, mask_positions.data,
                                                                        input_length,
                                                                        output_length, output_.data)
        event.wait()

    @staticmethod
    def set_input_at_mask_as_output(input_, mask_positions, input_length, output_length, output_):
        N, *input_shape = output_.shape
        device_global_shape = (N, int(np.prod(input_shape)))

        event = SplitInterfaceToDevice.device_program.set_input_at_mask_as_output(SplitInterfaceToDevice.device_queue,
                                                                                  device_global_shape, None,
                                                                                  input_.data, mask_positions.data,
                                                                                  input_length,
                                                                                  output_length, output_.data)
        event.wait()


class SplitChild(NetworkNode, LayerBaseAttributes, Layer):
    def __init__(self, mask):
        LayerBaseAttributes.__init__(self)
        NetworkNode.__init__(self)

        self.mask = mask
        self.mask_positions_device = None

    def build(self, device_context, device_queue):
        self.device_context = device_context
        self.device_queue = device_queue

        if not SplitInterfaceToDevice.initialized:
            SplitInterfaceToDevice(device_context, device_queue)

        input_shape = self.parents[0].output_shape

        self.input_shape = input_shape
        self.output_shape = (int(np.sum(self.mask)),)

        mask_positions = np.arange(int(np.prod(input_shape))).reshape(input_shape)[self.mask]
        self.mask_positions_device = cl_array.to_device(device_queue, mask_positions.astype(np.int32))

        self.built = True

    @check_built
    def predict(self, z, output_only=True, pre_activation_of_input=None, **kwargs):
        N = len(z)
        z_at_mask = cl_array.empty(self.device_queue, (N, *self.output_shape), dtype=np.float32)

        SplitInterfaceToDevice.get_input_at_mask(z, self.mask_positions_device, self.input_length_device,
                                                 self.output_length_device, z_at_mask)

        if output_only:
            return z_at_mask

        pre_activation_of_input_at_mask = cl_array.empty(self.device_queue, (N, *self.output_shape), dtype=np.float32)
        SplitInterfaceToDevice.get_input_at_mask(pre_activation_of_input, self.mask_positions_device,
                                                 self.input_length_device, self.output_length_device,
                                                 pre_activation_of_input_at_mask)

        return pre_activation_of_input_at_mask, z_at_mask

    @check_built
    def get_delta_backprop_(self, g_prime, new_delta, prev_z):
        summed_delta_device = reduce(lambda x, y: x + y, new_delta)
        return summed_delta_device

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
        return 'SplitChild Layer', f'Output Shape {(None, *self.output_shape)}'

    @property
    def activation_function_(self):
        return self.parents[0].activation_function_


class SplitLeftChild(SplitChild):
    """ There is no difference between the implementation of the Left and Right Child nodes.
        But these classes are created to make it clear which is the left and right node when
        looking at the repr of the instances
    """
    def __init__(self, mask):
        super().__init__(mask)


class SplitRightChild(SplitChild):
    """ There is no difference between the implementation of the Left and Right Child nodes.
        But these classes are created to make it clear which is the left and right node when
        looking at the repr of the instances
    """
    def __init__(self, mask):
        super().__init__(mask)


class Split(NetworkNode, LayerBaseAttributes, Layer):
    def __init__(self, mask):
        NetworkNode.__init__(self)
        LayerBaseAttributes.__init__(self)
        self.mask = mask.astype(bool)

        SplitLeftChild(self.mask)(self)
        SplitRightChild(~self.mask)(self)

    def build(self, device_context, device_queue):
        """ Initialise the layer
            Notes
            -----
            The output_shape is the same as the input_shape because the input must
            be based onto the children for it to be split
        """
        self.device_context = device_context
        self.device_queue = device_queue

        if not SplitInterfaceToDevice.initialized:
            SplitInterfaceToDevice(device_context, device_queue)

        input_shape = self.parents[0].output_shape

        self.input_shape = input_shape
        self.output_shape = input_shape

        self.built = True

    @check_built
    def predict(self, z, output_only=True, pre_activation_of_input=None, **kwargs):
        if output_only:
            return z.reshape(-1, *self.output_shape)
        return pre_activation_of_input.reshape(-1, *self.input_shape), z.reshape(-1, *self.output_shape)

    @check_built
    def get_delta_backprop_(self, g_prime, new_delta, prev_z):
        N = len(new_delta[0])

        out_delta = cl_array.empty(self.device_queue, (N, *self.input_shape), dtype=np.float32)

        for i, child in enumerate(self.children):
            SplitInterfaceToDevice.set_input_at_mask_as_output(out_delta, child.mask_positions_device,
                                                               child.input_length_device, child.output_length_device,
                                                               new_delta[i])

        return out_delta

    @check_built
    def get_parameter_gradients_(self, new_delta, prev_z):
        return {}

    @check_built
    def update_parameters_(self, *args):
        pass

    @check_built
    def get_weights(self, as_dict=False):
        if as_dict:
            return {}
        return None

    @check_built
    def summary_(self):
        return 'Split Layer', f'Output Shape {(None, *self.output_shape)}'

    @property
    def left(self):
        return self.children[0]

    @property
    def right(self):
        return self.children[1]

    @property
    def activation_function_(self):
        return self.parents[0].activation_function_
