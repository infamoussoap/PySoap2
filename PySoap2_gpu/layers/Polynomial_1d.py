import numpy as np
from functools import reduce

import pyopencl as cl
import pyopencl.array as cl_array

from PySoap2_gpu.layers import Layer
from PySoap2_gpu.layers import NetworkNode
from PySoap2_gpu.layers.LayerBaseAttributes import LayerBaseAttributes

from .PolynomialTransformation import PolynomialTransformationInterface

from .ValueChecks import check_built


class Polynomial_1d(NetworkNode, LayerBaseAttributes, Layer):
    """ 1-d Polynomial transform the input to this layer

        Attributes
        ----------
        P1 : (M1, M1) cl_array
            Note that P1 is assumed such that P1.T @ P1 = Identity
        M1 : np.int32

        Notes
        -----
        The input to this layer is assumed to either be 1 dimensional, or 2 dimensional data.
    """
    def __init__(self, P1):
        """ P1 : (M1, M1) cl_array """
        NetworkNode.__init__(self)
        LayerBaseAttributes.__init__(self)

        self.P1 = P1
        self.M1 = None

    def build(self, device_context, device_queue):
        """ Initialises the weight and bias units """
        self.device_context = device_context
        self.device_queue = device_queue

        PolynomialTransformationInterface(device_context, device_queue)

        input_shape = self.parents[0].output_shape

        if len(input_shape) != 1 and len(input_shape) != 2:
            raise ValueError('Input to Kravchuk_1d assumed to either be 1d or 2d data, not '
                             f'{len(input_shape)}d')

        self.input_shape = input_shape
        self.output_shape = input_shape

        self.M1 = np.int32(input_shape[0])

        self.built = True

    @check_built
    def predict(self, z, output_only=True, **kwargs):
        out = cl_array.empty_like(z)
        PolynomialTransformationInterface.polynomial_transform_1d(self.P1, z, self.M1,
                                                                  self.input_length_device, out)

        if output_only:
            return out
        return out, out

    @check_built
    def get_delta_backprop_(self, g_prime, new_delta, prev_z):
        """ new_delta assumed to be a list of cl_array.Array """
        out = cl_array.empty_like(new_delta[0])
        delta = reduce(lambda x, y: x + y, new_delta)

        # Inverse Transform
        PolynomialTransformationInterface.polynomial_transform_1d(self.P1.T, delta, self.M1,
                                                                  self.input_length_device, out)

        return out

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
