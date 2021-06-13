import numpy as np

from PySoap2.layers import Layer
from PySoap2.layers.NetworkNode import NetworkNode
from PySoap2.layers.LayerBaseAttributes import LayerBaseAttributes

from .LayerBuiltChecks import check_built


class Reshape(NetworkNode, LayerBaseAttributes, Layer):
    def __init__(self, shape):
        """ Initialise the class

            Parameters
            ----------
            shape : tuple of int
        """
        NetworkNode.__init__(self)
        LayerBaseAttributes.__init__(self)

        if any([i < 0 for i in shape]):
            raise ValueError('Reshape does not support negative dimensions.')

        self.shape = shape
        self.activation_function = 'reshape'

    def build(self):
        """ Built/initialised the layer """
        input_shape = self.parents[0].output_shape

        if np.prod(input_shape).astype(int) != np.prod(self.shape).astype(int):
            raise ValueError(f'Cannot reshape input of shape {input_shape} into shape {self.shape}.')

        self.input_shape = input_shape
        self.output_shape = self.shape

        self.built = True

    @check_built
    def predict(self, z, output_only=True, pre_activation_of_input=None):
        if output_only:
            return z.reshape(-1, *self.output_shape)
        return pre_activation_of_input, z.reshape(-1, *self.output_shape)

    @check_built
    def get_delta_backprop_(self, g_prime, new_delta, *args, **kwargs):
        delta = np.sum(np.array(new_delta), axis=0)
        return delta.reshape(-1, *self.input_shape)

    @check_built
    def get_parameter_gradients_(self, *args, **kwargs):
        return {}

    @check_built
    def update_parameters_(self, parameter_gradients):
        pass

    @check_built
    def get_weights(self, as_dict=False):
        if as_dict:
            return {}
        return None

    @check_built
    def summary_(self):
        return f'Reshape', f'Output Shape {(None, *self.output_shape)}'

    @property
    def activation_function_(self):
        def reshaped_activation_function(x, grad=False):
            parent = self.parents[0]
            post_activation = parent.activation_function_(x, grad=grad)

            if parent.activation_function == 'linear' and grad:
                return post_activation

            return self.predict(post_activation, output_only=True)

        return reshaped_activation_function
