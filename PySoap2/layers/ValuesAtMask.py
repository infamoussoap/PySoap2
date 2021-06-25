import numpy as np

from PySoap2.layers import Layer
from PySoap2.layers.NetworkNode import NetworkNode
from PySoap2.layers.LayerBaseAttributes import LayerBaseAttributes

from .LayerBuiltChecks import check_built


class ValuesAtMask(NetworkNode, LayerBaseAttributes, Layer):
    """ Given an input to this layer, this will only return the values at the mask positions """
    def __init__(self, mask):
        NetworkNode.__init__(self)
        LayerBaseAttributes.__init__(self)

        self.mask = mask.astype(bool)

    def build(self):
        self.input_shape = self.parents[0].output_shape
        self.output_shape = (np.sum(self.mask),)

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

        if output_only:
            return z[:, self.mask]
        return pre_activation_of_input[:, self.mask], z[:, self.mask]

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
        delta = np.sum(np.array(new_delta), axis=0)
        N = len(delta)

        buffered_delta = np.zeros((N, *self.input_shape))
        buffered_delta[:, self.mask] = delta

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
