import numpy as np

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
        self.input_shape = self.parents[0].output_shape
        self.output_shape = self.input_shape
        self.built = True

    @check_built
    def predict(self, z, output_only=True):
        """ Returns the output of this layer

            Parameters
            ----------
            z : np.array
                The output of the parent node
            output_only : bool
                If output_only is True then only the output (concatenation) will return
                Otherwise the tuple of inputs and concatenation will be returned

            Notes
            -----
            There is no concatenation that happens in this layer, as the concatenation
            requires the knowledge of all the parent nodes of the Concatenate node. As such,
            concatenation occurs in the Concatenate node.
        """
        if output_only:
            return z
        return z, z

    @check_built
    def get_delta_backprop_(self, g_prime, new_delta, *args):
        """ Returns delta^{k-1}

            Parameters
            ----------
            g_prime : (N, ...) np.array
                Should be the derivative of the output of the previous layer, g'_{k-1}
            new_delta : (N, ...) np.array
                The delta for this layer, delta^k

            Returns
            -------
            (N, *self.input_shape) np.array
        """
        return new_delta[:, self.mask].reshape((len(new_delta), *self.input_shape))

    @check_built
    def get_parameter_gradients_(self, delta, prev_z):
        return {}

    @check_built
    def update_parameters_(self, parameter_updates):
        pass

    @check_built
    def get_weights(self):
        return None

    @check_built
    def summary_(self):
        return f'Concat-Parent', f'Output Shape {(None, *self.output_shape)}'


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
        self.input_shape = tuple([parent.output_shape for parent in self.parents])

        if not self._is_concat_valid(list(self.input_shape), self.axis):
            raise ValueError('Every dimension, except for the concatenation axis, must be equal.')

        self.output_shape = self._concat_shape(list(self.input_shape), self.axis)

        # The mappings from the inputs and their position in the concatenated output
        for i in range(len(self.input_shape)):
            new_mask = [np.ones(shape) if i == j else np.zeros(shape) for (j, shape) in enumerate(self.input_shape)]
            self.parents[i].mask = np.concatenate(new_mask, axis=self.axis).astype(bool)

        self.built = True

    @check_built
    def predict(self, z, output_only=True):
        """ Forward propagation of this layer

            Parameters
            ----------
            z : tuple of np.array
                The inputs to be concatenated. Note that the order of the tuple should be in the same
                order as given by the parents attribute
            output_only : bool
                If output_only is True then only the output (concatenation) will return
                Otherwise the tuple of inputs and concatenation will be returned
        """
        if output_only:
            return np.concatenate(z, axis=self.axis)
        return z, np.concatenate(z, axis=self.axis)

    @check_built
    def get_delta_backprop_(self, g_prime, new_delta, *args):
        """ Returns delta^{k-1}

            Parameters
            ----------
            g_prime : np.array
                Should be the derivative of the ouput of the previous layer, g'_{k-1}(a^{k-1}_{m,j})
            new_delta : np.array
                The delta for this layer, delta^k_{m, j}

            Returns
            -------
            np.array

            Notes
            -----
            The Concatenate class does not return the unconcatenated delta. That is performed by the
            ConcatenateParent, allowing each instance of the Concatenate and ConcatenateParent class
            to have a static output shape of delta^{k-1}
        """
        return new_delta

    @check_built
    def get_parameter_gradients_(self, delta, prev_z):
        return {}

    @check_built
    def update_parameters_(self, parameter_updates):
        pass

    @check_built
    def get_weights(self):
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

