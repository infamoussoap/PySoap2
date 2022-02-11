import numpy as np
from functools import reduce

import pyopencl.array as cl_array

from PySoap2_gpu.layers import Layer
from PySoap2_gpu.layers import NetworkNode
from PySoap2_gpu.layers.LayerBaseAttributes import LayerBaseAttributes

from .ValueChecks import check_built

from PySoap2_gpu.layers.ProgramInterface.SplitInterface import SplitInterface


class SplitChild(NetworkNode, LayerBaseAttributes, Layer):
    def __init__(self, mask):
        LayerBaseAttributes.__init__(self)
        NetworkNode.__init__(self)

        self.mask = mask
        self.mask_positions_device = None

    def build(self, device_context, device_queue):
        self.context = device_context
        self.queue = device_queue

        if not SplitInterface.initialized:
            SplitInterface(device_context, device_queue)

        input_shape = self.parents[0].output_shape

        self.input_shape = input_shape
        self.output_shape = (int(np.sum(self.mask)),)

        mask_positions = np.arange(int(np.prod(input_shape))).reshape(input_shape)[self.mask]
        self.mask_positions_device = cl_array.to_device(device_queue, mask_positions.astype(np.int32))

        self.built = True

    @check_built
    def predict(self, z, output_only=True, pre_activation_of_input=None, **kwargs):
        z_at_mask = SplitInterface.get_input_at_mask(z, self.mask_positions_device, self.output_shape)

        if output_only:
            return z_at_mask

        pre_activation_of_input_at_mask = SplitInterface.get_input_at_mask(pre_activation_of_input,
                                                                           self.mask_positions_device,
                                                                           self.output_shape)

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
        self.context = device_context
        self.queue = device_queue

        if not SplitInterface.initialized:
            SplitInterface(device_context, device_queue)

        input_shape = self.parents[0].output_shape

        self.input_shape = input_shape
        self.output_shape = input_shape

        if self.input_shape != self.mask.shape:
            raise ValueError(f'Mask shape {self.mask.shape} is not the same as input shape {self.input_shape}.')

        self.built = True

    @check_built
    def predict(self, z, output_only=True, pre_activation_of_input=None, **kwargs):
        if output_only:
            return z.reshape(-1, *self.output_shape)
        return pre_activation_of_input.reshape(-1, *self.input_shape), z.reshape(-1, *self.output_shape)

    @check_built
    def get_delta_backprop_(self, g_prime, new_delta, prev_z):
        N = len(new_delta[0])

        out_delta = cl_array.empty(self.queue, (N, *self.input_shape), dtype=np.float64)

        for i, child in enumerate(self.children):
            SplitInterface.set_input_at_mask_as_output(out_delta, child.mask_positions_device,
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
