from PySoap2_gpu.layers import Layer
from PySoap2_gpu.layers.NetworkNode import NetworkNode
from PySoap2_gpu.layers.LayerBaseAttributes import LayerBaseAttributes

from .ValueChecks import assert_instance_of_cl_array


class Input(NetworkNode, LayerBaseAttributes, Layer):
    def __init__(self, input_shape):
        """ Creates a `Input` class with a given input shape

            Parameters
            ----------
            input_shape : tuple of int
                The shape of a given data point
        """
        NetworkNode.__init__(self)
        LayerBaseAttributes.__init__(self)

        self.input_shape = (*input_shape, )
        self.output_shape = (*input_shape, )

        # This layer is assumed to be built from creation
        self.built = True

    def build(self, device_context, device_queue):
        """ Initialises the layer

            Notes
            -----
            Since this is simply a pass through layer, there is no initialization needed.
            This method is only written so as to make the `Layer` uniform in
            implementation
        """
        self.device_context = device_context
        self.device_queue = device_queue

    def predict(self, z, *args, output_only=True, **kwargs):
        assert_instance_of_cl_array(z)

        if output_only:
            return z
        return z, z

    def get_delta_backprop_(self, g_prime, new_delta, *args):
        return None

    def get_parameter_gradients_(self, delta, prev_z):
        return {}

    def update_parameters_(self, parameter_updates):
        pass

    def get_weights(self):
        return None, None

    def summary_(self):
        return f'Input', f'Input Shape  {(None, *self.input_shape)}'

    def __str__(self):
        return f'Input: Input Shape  {(None, *self.input_shape)}'
