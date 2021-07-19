import numpy as np
from functools import reduce

import pyopencl as cl
import pyopencl.array as cl_array

from PySoap2_gpu.layers import Layer
from PySoap2_gpu.layers import NetworkNode
from PySoap2_gpu.layers.LayerBaseAttributes import LayerBaseAttributes

from .Split import SplitInterfaceToDevice

from .ValueChecks import check_built


class ValuesAtMask(NetworkNode, LayerBaseAttributes, Layer):
    """ Given an input to this layer, this will only return the values at the mask positions """

    def __init__(self, mask):
        NetworkNode.__init__(self)
        LayerBaseAttributes.__init__(self)

        self.mask = mask.astype(bool)
        self.mask_positions_device = None

    def build(self, device_context, device_queue):
        self.device_context = device_context
        self.device_queue = device_queue

        self.input_shape = self.parents[0].output_shape
        self.output_shape = (np.sum(self.mask),)

        mask_positions = np.arange(int(np.prod(self.input_shape))).reshape(self.input_shape)[self.mask]
        self.mask_positions_device = cl_array.to_device(device_queue, mask_positions.astype(np.int32))

        if not SplitInterfaceToDevice.initialized:
            SplitInterfaceToDevice(self.device_context, self.device_queue)

        self.built = True

    @check_built
    def predict(self, z, output_only=True, pre_activation_of_input=None, **kwargs):
        """ Forward propagate the input at the mask positions

            Parameters
            ----------
            z : (N, *input_shape) np.array
            output_only : bool, optional
            pre_activation_of_input : (N, *input_shape) np.array

            Returns
            -------
            (N, i) np.array
                The split if output_only=True
            OR
            (N, *input_shape) np.array, (N, i) np.array
                If output_only=False, the first position will be the original input
                with the second being the split of the input
        """

        N = len(z)
        z_at_mask = cl_array.empty(self.device_queue, (N, *self.output_shape), dtype=np.float64)
        SplitInterfaceToDevice.get_input_at_mask(z, self.mask_positions_device, self.input_length_device,
                                                 self.output_length_device, z_at_mask)

        if output_only:
            return z_at_mask

        pre_activation_of_input_at_mask = cl_array.empty(self.device_queue, (N, *self.output_shape), dtype=np.float64)
        SplitInterfaceToDevice.get_input_at_mask(pre_activation_of_input, self.mask_positions_device,
                                                 self.input_length_device, self.output_length_device,
                                                 pre_activation_of_input_at_mask)

        return pre_activation_of_input_at_mask, z_at_mask

    @check_built
    def get_delta_backprop_(self, g_prime, new_delta, *args, **kwargs):
        """ Returns delta^{k-1}

            Parameters
            ----------
            g_prime : (N, *input_shape) np.array
                Should be g'_{k-1}
            new_delta : list of (N, *output_shape) np.array

            Returns
            -------
            (N, *input_shape)

            Notes
            -----
            This layer essential performs a mapping from the input into a subset
            of the input. As such, to find delta, we essentially need to do the reverse
            mapping and buffer everything to zero.
        """
        delta = reduce(lambda x, y: x + y, new_delta)
        N = len(delta)

        buffered_delta = cl_array.zeros(self.device_queue, (N, *self.input_shape), dtype=np.float64)

        SplitInterfaceToDevice.set_input_at_mask_as_output(buffered_delta, self.mask_positions_device,
                                                           self.input_length_device, self.output_length_device,
                                                           delta)

        return buffered_delta

    @check_built
    def get_parameter_gradients_(self, new_delta, prev_z):
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
        return 'ValueAtMask Layer', f'Output Shape {(None, *self.output_shape)}'

    @property
    def activation_function_(self):
        return self.parents[0].activation_function_
