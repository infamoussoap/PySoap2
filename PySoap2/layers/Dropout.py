import numpy as np
from functools import reduce

from PySoap2.layers import Layer
from PySoap2.layers.NetworkNode import NetworkNode
from PySoap2.layers.LayerBaseAttributes import LayerBaseAttributes

from .LayerBuiltChecks import check_built


class Dropout(NetworkNode, LayerBaseAttributes, Layer):
    def __init__(self, rate):
        """ rate : Float between 0 and 1. The fraction of inputs to drop """
        NetworkNode.__init__(self)
        LayerBaseAttributes.__init__(self)

        self.rate = rate
        self.mask = None

    def build(self):
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

        self.mask = np.random.rand(*self.output_shape) > self.rate

        # The inverted dropout method, where scaling is performed during training, so the
        # forward pass, during testing, does not need to be scaled.
        # see https://cs231n.github.io/neural-networks-2/

        z_inverted_dropout = z * self.mask / (1 - self.rate)
        if output_only:
            return z_inverted_dropout
        return z_inverted_dropout, z_inverted_dropout

    @check_built
    def get_delta_backprop_(self, g_prime, new_delta, prev_z):
        delta = reduce(lambda x, y: x + y, new_delta)
        return delta * self.mask

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
