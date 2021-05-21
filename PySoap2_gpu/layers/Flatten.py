import numpy as np
from functools import reduce

from PySoap2_gpu.layers import Layer
from PySoap2_gpu.layers.NetworkNode import NetworkNode
from PySoap2_gpu.layers.LayerBaseAttributes import LayerBaseAttributes

from .ValueChecks import assert_instance_of_cl_array
from .ValueChecks import check_built


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

    @check_built
    def predict(self, z, output_only=True, pre_activation_of_input=None):
        assert_instance_of_cl_array(z)

        if output_only:
            return z.reshape(-1, *self.output_shape)
        return pre_activation_of_input, z.reshape(-1, *self.output_shape)

    @check_built
    def get_delta_backprop_(self, g_prime, new_delta, *args, **kwargs):
        assert_instance_of_cl_array(g_prime)

        summed_delta_device = reduce(lambda x, y: x + y, new_delta)

        return summed_delta_device.reshape(-1, *self.input_shape)

    @check_built
    def get_parameter_gradients_(self, *args, **kwargs):
        return {}

    @check_built
    def update_parameters_(self, parameter_gradients):
        pass

    @check_built
    def get_weights(self):
        return None, None

    @check_built
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
