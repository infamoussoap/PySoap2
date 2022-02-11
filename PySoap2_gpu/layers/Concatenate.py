import numpy as np
from functools import reduce

import pyopencl.array as cl_array

from PySoap2_gpu.layers import Layer
from PySoap2_gpu.layers import NetworkNode
from PySoap2_gpu.layers.LayerBaseAttributes import LayerBaseAttributes

from .Split import SplitInterface
from .ValueChecks import check_built


class ConcatenateParent(NetworkNode, LayerBaseAttributes, Layer):
    def __init__(self):
        NetworkNode.__init__(self)
        LayerBaseAttributes.__init__(self)

        self.mask_positions_device = None  # To be determined by concat child

    def build(self, device_context, device_queue):
        self.context = device_context
        self.queue = device_queue

        if not SplitInterface.initialized:
            SplitInterface(device_context, device_queue)

        input_shape = self.parents[0].output_shape

        self.input_shape = input_shape
        self.output_shape = None  # To be determined by the concat child

        self.built = True

    @check_built
    def predict(self, z, output_only=True, pre_activation_of_input=None, **kwargs):
        if output_only:
            return z
        return pre_activation_of_input, z

    @check_built
    def get_delta_backprop_(self, g_prime, new_delta, prev_z):
        """ new_delta is assumed to come from the concatenate layer, and so is a list of
            1 cl_array
         """
        delta = new_delta[0]

        out = SplitInterface.get_input_at_mask(delta, self.mask_positions_device, self.input_shape)

        return out

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
        return f'Concat-Parent', f'Output Shape {(None, *self.output_shape)}'

    @property
    def activation_function_(self):
        return self.parents[0].activation_function_


class Concatenate(NetworkNode, LayerBaseAttributes, Layer):

    @staticmethod
    def _is_concat_valid(shape_list, axis):
        """ Checks to see if the given concatenation is valid. Concatenation is only valid if every dimension,
            except for the concatenation axis, is equal.

            Parameters
            ----------
            shape_list : list of tuple
                List of array shapes
            axis : int
                The axis for concatenation
        """
        if len(shape_list) == 1:
            return True

        shape_zeroed_axis_list = [list(shape) for shape in shape_list]
        for shape in shape_zeroed_axis_list:
            shape[axis] = 0

        valid_shape = shape_zeroed_axis_list[0]
        return all([shape == valid_shape for shape in shape_zeroed_axis_list])

    @staticmethod
    def _concat_shape(shape_list, axis):
        """ Returns the final concatenation shape of the inputs

            Parameters
            ----------
            shape_list : list of tuple
                List of array shapes
            axis : int
                The axis for concatenation

            Raises
            ------
            ValueError
                If the concatenation is not valid
        """

        if Concatenate._is_concat_valid(shape_list, axis):
            new_shape = list(shape_list[0])
            new_shape[axis] = np.sum([shape[axis] for shape in shape_list])
            return tuple(new_shape)

        raise ValueError('Every dimension, except for the concatenation axis, must be equal.')

    def __init__(self, axis=-1):
        NetworkNode.__init__(self)
        LayerBaseAttributes.__init__(self)

        self.axis = axis

    def build(self, device_context, device_queue):
        """ Initialise the input and output shape

            Raises
            ------
            ValueError
                If the concatenation is not valid
        """
        self.context = device_context
        self.queue = device_queue

        input_shape_of_concat_parents = [parent.input_shape for parent in self.parents]

        if not self._is_concat_valid(input_shape_of_concat_parents, self.axis):
            raise ValueError('Every dimension, except for the concatenation axis, must be equal.')

        self.input_shape = tuple(input_shape_of_concat_parents)
        self.output_shape = self._concat_shape(input_shape_of_concat_parents, self.axis)

        for concat_parent in self.parents:
            concat_parent.output_shape = self.output_shape
            concat_parent.built = True

        # The mappings from the inputs and their position in the concatenated output
        positions = np.arange(int(np.prod(self.output_shape))).reshape(*self.output_shape)

        for i in range(len(input_shape_of_concat_parents)):
            new_mask = [np.ones(shape) if i == j else np.zeros(shape)
                        for (j, shape) in enumerate(input_shape_of_concat_parents)]
            mask = np.concatenate(new_mask, axis=self.axis).astype(bool)
            mask_positions = (positions[mask]).astype(np.int32)
            self.parents[i].mask_positions_device = cl_array.to_device(self.queue, mask_positions)

        self.built = True

    @check_built
    def predict(self, z, output_only=True, pre_activation_of_input=None, **kwargs):
        if output_only:
            return self._concatenate_with_parents_mask_positions(z)

        return (self._concatenate_with_parents_mask_positions(pre_activation_of_input),
                self._concatenate_with_parents_mask_positions(z))

    def _concatenate_with_parents_mask_positions(self, z):
        """ z is assumed to be a list of cl_arrays """
        mask_positions_device = [parent.mask_positions_device for parent in self.parents]

        return self.concatenate_with_mask_positions(self.queue, z, mask_positions_device, self.output_shape)

    @staticmethod
    def concatenate_with_mask_positions(device_queue, z_device, mask_positions_device, output_shape):
        N = len(z_device[0])
        input_ = cl_array.empty(device_queue, (N, *output_shape), dtype=np.float64)

        output_length_device = np.int32(np.prod(output_shape))
        for array, mask_positions in zip(z_device, mask_positions_device):
            input_shape = array.shape[1:]
            input_length_device = np.int32(np.prod(input_shape))

            SplitInterface.set_input_at_mask_as_output(input_, mask_positions,
                                                       output_length_device,
                                                       input_length_device, array)
        return input_

    @check_built
    def get_delta_backprop_(self, g_prime, new_delta, prev_z):
        return reduce(lambda x, y: x + y, new_delta)

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
        return 'Concatenate', f'Output Shape: {(None, *self.output_shape)}'

    def __call__(self, input_layers):
        """ __call__ of NetworkNode is overloaded as it is now assumed the parameter
            is a list of Layer

            Parameters
            ----------
            input_layers : list of Layer
        """
        concatenate_parent_nodes = [ConcatenateParent()(layer) for layer in input_layers]
        for parent_node in concatenate_parent_nodes:
            parent_node.add_child(self)

        self.add_parents(tuple(concatenate_parent_nodes))

        return self

    @property
    def activation_function_(self):
        return self.parents[0].activation_function_


def set_list_of_inputs_at_masks_as_outputs(device_queue, inputs_, mask_positions, output_shape):
    """ asd

        Parameters
        ----------
        device_queue : cl.CommandQueue
        inputs_ : list of (N, ...) cl_array.Array
            The that each cl_array in the list don't need to have the same dimensions. Only that
            the first dimension is the same (the first dimension is the length of the dataset)
        mask_positions : list of (N, ...) cl_array.Array
            The that each cl_array in the list don't need to have the same dimensions. Only that
            the first dimension is the same (the first dimension is the length of the dataset)
        output_shape : tuple of int
    """

    N = len(inputs_[0])
    input_ = cl_array.empty(device_queue, (N, *output_shape), dtype=np.float64)

    output_length_device = np.int32(np.prod(output_shape))

    for array, mask_positions in zip(inputs_, mask_positions):
        input_shape = array.shape[1:]
        input_length_device = np.int32(np.prod(input_shape))

        SplitInterface.set_input_at_mask_as_output(input_, mask_positions,
                                                   output_length_device,
                                                   input_length_device, array)
    return input_
