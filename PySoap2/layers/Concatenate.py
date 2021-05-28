import numpy as np
from functools import reduce

from PySoap2.layers import Layer
from PySoap2.layers.NetworkNode import NetworkNode
from PySoap2.layers.LayerBaseAttributes import LayerBaseAttributes

from .LayerBuiltChecks import check_built


class ConcatenateParent(NetworkNode, LayerBaseAttributes, Layer):
    """ Parent of Concatenate Node. This class provides an interface between the parent
        nodes to be concatenated and the Concatenate class.

        Attributes
        ----------
        mask : np.array of bool
            The mapping of the input and its position in the concatenated output
    """
    def __init__(self):
        """ Initialise Class """
        NetworkNode.__init__(self)
        LayerBaseAttributes.__init__(self)

        self.mask = None

    def build(self):
        """ Set the input shape for this layer

            Notes
            -----
            The output shape is determined by the concatenate (child) layer. It is in that layer
            where this class will be considered built.
        """

        self.input_shape = self.parents[0].output_shape
        self.built = False

    @check_built
    def predict(self, z, output_only=True, pre_activation_of_input=None):
        """ Returns the output of this layer

            Parameters
            ----------
            z : np.array
                The output of the parent node
            output_only : bool
                If output_only is True then only the output (concatenation) will return
                Otherwise the tuple of inputs and concatenation will be returned
            pre_activation_of_input : (N, *input_shape) np.array
                The input, z, before it passed through the activation function

            Notes
            -----
            There is no concatenation that happens in this layer, as the concatenation
            requires the knowledge of all the parent nodes of the Concatenate node. As such,
            concatenation occurs in the Concatenate node.
        """
        buffered_z = np.zeros((len(z), *self.output_shape))
        buffered_z[:, self.mask] = z

        if output_only:
            return buffered_z

        buffered_pre_activation = np.zeros((len(z), *self.output_shape))
        buffered_pre_activation[:, self.mask] = pre_activation_of_input

        return buffered_pre_activation, buffered_z

    @check_built
    def get_delta_backprop_(self, g_prime, new_delta, *args):
        """ Returns delta^{k-1}

            Parameters
            ----------
            g_prime : (N, ...) np.array
            new_delta : list of (N, ...) np.array
                Note that ConcatenateParent is only assumed to have 1 child, the Concatenate layer.
                So this is a list of only 1 np.array

            Returns
            -------
            (N, *self.input_shape) np.array
        """
        return new_delta[0][:, self.mask].reshape((-1, *self.input_shape))

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

    def build(self):
        """ Initialise the input and output shape

            Raises
            ------
            ValueError
                If the concatenation is not valid
        """
        input_shape_of_concat_parents = tuple([parent.input_shape for parent in self.parents])

        if not self._is_concat_valid(list(input_shape_of_concat_parents), self.axis):
            raise ValueError('Every dimension, except for the concatenation axis, must be equal.')

        self.output_shape = self._concat_shape(list(input_shape_of_concat_parents), self.axis)
        self.input_shape = tuple([self.output_shape]*len(self.parents))

        for parent in self.parents:
            parent.output_shape = self.output_shape
            parent.built = True

        # The mappings from the inputs and their position in the concatenated output
        for i in range(len(input_shape_of_concat_parents)):
            new_mask = [np.ones(shape) if i == j else np.zeros(shape)
                        for (j, shape) in enumerate(input_shape_of_concat_parents)]
            self.parents[i].mask = np.concatenate(new_mask, axis=self.axis).astype(bool)

        self.built = True

    @check_built
    def predict(self, z, output_only=True, pre_activation_of_input=None):
        """ Forward propagation of this layer

            Parameters
            ----------
            z : tuple of np.array
                The inputs to be concatenated. Note that the order of the tuple should be in the same
                order as given by the parents attribute
            output_only : bool
                If output_only is True then only the output (concatenation) will return
                Otherwise the tuple of inputs and concatenation will be returned
            pre_activation_of_input : (N, *input_shape) np.array
                The input, z, before it passed through the activation function
        """

        output_at_mask = [parent_output[:, parent.mask] for (parent, parent_output) in zip(self.parents, z)]
        if output_only:
            return np.concatenate(output_at_mask, axis=self.axis)

        pre_activation_at_mask = [pre_activation_of_parent[:, parent.mask]
                                  for (parent, pre_activation_of_parent) in zip(self.parents, pre_activation_of_input)]
        return np.concatenate(pre_activation_at_mask, axis=self.axis), np.concatenate(output_at_mask, axis=self.axis)

    @check_built
    def get_delta_backprop_(self, g_prime, new_delta, *args):
        """ Returns delta^{k-1}

            Parameters
            ----------
            g_prime : np.array
            new_delta : list of np.array
                While the ConcatenateParent can only have one layer, the concatenate node can have multiple

            Returns
            -------
            np.array

            Notes
            -----
            The Concatenate class does not return the unconcatenated delta. That is performed by the
            ConcatenateParent, allowing each instance of the Concatenate and ConcatenateParent class
            to have a static output shape of delta^{k-1}
        """
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
