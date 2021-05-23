from functools import reduce

from PySoap2.layers import Layer
from PySoap2.layers.NetworkNode import NetworkNode
from PySoap2.layers.LayerBaseAttributes import LayerBaseAttributes

from .PolynomialTransformations import KravchukTransform

from .LayerBuiltChecks import check_built


class Kravchuk_2d(NetworkNode, LayerBaseAttributes, Layer):
    """ 2-d Kravchuk transform the input to this layer

        Notes
        -----
        The input to this layer is assumed to either be 2 dimensional, or 3 dimensional data.
    """
    def __init__(self, p=0.5):
        NetworkNode.__init__(self)
        LayerBaseAttributes.__init__(self)

        self.p = p

    def build(self):
        """ Initialises the weight and bias units """
        input_shape = self.parents[0].output_shape

        if len(input_shape) != 2 and len(input_shape) != 3:
            raise ValueError('Input to Kravchuk_2d assumed to either be 2d or 3d data, not '
                             f'{len(input_shape)}d')

        self.input_shape = input_shape
        self.output_shape = input_shape

        self.built = True

    @check_built
    def predict(self, z, output_only=True, pre_activation_of_input=None):
        transformed_z = KravchukTransform.transform_2d(z, p=self.p)

        if output_only:
            return transformed_z
        return transformed_z, transformed_z

    @check_built
    def get_delta_backprop_(self, g_prime, new_delta, prev_z):
        delta = reduce(lambda x, y: x + y, new_delta)
        return KravchukTransform.transform_2d(delta, p=self.p, inverse=True)

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
        return 'Kravchuk Transformation', f'Output Shape {self.input_shape}'
