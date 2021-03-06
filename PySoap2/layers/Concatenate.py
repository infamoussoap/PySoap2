import numpy as np

from PySoap2.layers import Layer
from PySoap2.layers.NetworkNode import NetworkNode


class Concatenate(NetworkNode, Layer):
    def __init__(self, axis=-1):
        NetworkNode.__init__(self)

        self.axis = axis

        self.built = False

    def build(self, input_shape):
        pass

    def predict(self, z, output_only=True):
        pass

    def get_delta_backprop_(self, g_prime, new_delta, prev_z):
        pass

    def get_parameter_gradients_(self, delta, prev_z):
        pass

    def update_parameters_(self, parameter_updates):
        pass

    def get_weights(self):
        pass

    def summary_(self):
        pass

    def activation_function_(self):
        pass
