from functools import reduce

from PySoap2.layers import Layer
from PySoap2.layers.NetworkNode import NetworkNode
from PySoap2.layers.LayerBaseAttributes import LayerBaseAttributes

from .PolynomialTransformations import PolynomialTransform

from .LayerBuiltChecks import check_built


class Polynomial_2d(NetworkNode, LayerBaseAttributes, Layer):
    """ 2-d Polynomial transform the input to this layer

        Attributes
        ----------
        P1 : (i, i) np.array
            Note that for the back-prop to work correctly, it is assumed that P2.T @ P2 = identity
        P2 : (j, j) np.array, optional
            If P2 is None, then it is assumed that P2 = P1
            Note that for the back-prop to work correctly, it is assumed that P2.T @ P2 = identity

        Notes
        -----
        The input to this layer is assumed to either be 2 dimensional, or 3 dimensional data.
    """
    def __init__(self, P1, P2=None):
        NetworkNode.__init__(self)
        LayerBaseAttributes.__init__(self)

        self.P1 = P1
        self.P2 = P2

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
        transformed_z = PolynomialTransform.transform_2d(z, self.P1, self.P2)

        if output_only:
            return transformed_z
        return transformed_z, transformed_z

    @check_built
    def get_delta_backprop_(self, g_prime, new_delta, prev_z):
        delta = reduce(lambda x, y: x + y, new_delta)
        return PolynomialTransform.transform_2d(delta, self.P1.T, self.P2.T)

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
        return 'Polynomial Transformation', f'Output Shape {self.input_shape}'
