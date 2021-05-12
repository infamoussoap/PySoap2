import numpy as np

from PySoap2_gpu.layers import Layer
from PySoap2_gpu.layers.NetworkNode import NetworkNode
from PySoap2_gpu.layers.LayerBaseAttributes import LayerBaseAttributes

from .ValueChecks import assert_instance_of_cl_array


class Flatten(NetworkNode, LayerBaseAttributes, Layer):
    def __init__(self):
        NetworkNode.__init__(self)
        LayerBaseAttributes.__init__(self)

    def build(self, device_context, device_queue):
        """ Built/initialised the layer """
        self.device_context = device_context
        self.device_queue = device_queue

        input_shape = self.parents[0].output_shape

        self.input_shape = input_shape
        self.output_shape = (np.prod(input_shape),)

        self.built = True

    def predict(self, z, output_only=True, pre_activation_of_input=None):
        assert_instance_of_cl_array(z)

        if output_only:
            return z.reshape(-1, *self.output_shape)
        return pre_activation_of_input, z.reshape(-1, *self.output_shape)

    def get_delta_backprop_(self, g_prime, new_delta, *args, **kwargs):
        assert_instance_of_cl_array(g_prime)
        assert_instance_of_cl_array(new_delta)

        return new_delta.reshape(-1, *self.input_shape)

    def get_parameter_gradients_(self, *args, **kwargs):
        return {}

    def update_parameters_(self, parameter_gradients):
        pass

    def get_weights(self):
        return None, None

    def summary_(self):
        return f'Flatten', f'Output Shape {(None, *self.output_shape)}'

    def __str__(self):
        return f'Flatten'

    @property
    def activation_function_(self):
        def reshaped_activation_function(x, grad=False):
            parent = self.parents[0]
            post_activation = parent.activation_function_(x, grad=grad)

            return self.predict(post_activation, output_only=True)

        return reshaped_activation_function
