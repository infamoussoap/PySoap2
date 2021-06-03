import numpy as np
from functools import reduce

import pyopencl as cl
import pyopencl.array as cl_array

from PySoap2.layers.PolynomialTransformations.KravchukTransform import kravchuk_polynomials

from PySoap2_gpu.layers import Layer
from PySoap2_gpu.layers.NetworkNode import NetworkNode
from PySoap2_gpu.layers.LayerBaseAttributes import LayerBaseAttributes

from .PolynomialTransformation import PolynomialTransformationInterface

from .ValueChecks import check_built


class Kravchuk_1d(NetworkNode, LayerBaseAttributes, Layer):
    """ 1-d Kravchuk transform the input to this layer

        Notes
        -----
        The input to this layer is assumed to either be 1 dimensional, or 2 dimensional data.
    """
    def __init__(self, p=0.5):
        NetworkNode.__init__(self)
        LayerBaseAttributes.__init__(self)

        self.p = p
        self.P1_device = None  # The kravchuk polynomial
        self.M1 = None

    def build(self, device_context, device_queue):
        """ Initialises the weight and bias units """
        self.device_context = device_context
        self.device_queue = device_queue

        if not PolynomialTransformationInterface.initialize:
            PolynomialTransformationInterface(device_context, device_queue)

        input_shape = self.parents[0].output_shape

        if len(input_shape) != 1 and len(input_shape) != 2:
            raise ValueError('Input to Kravchuk_1d assumed to either be 1d or 2d data, not '
                             f'{len(input_shape)}d')

        self.input_shape = input_shape
        self.output_shape = input_shape

        self.M1 = np.int32(input_shape[0])
        P1 = kravchuk_polynomials(self.M1 - 1, self.p).astype(np.float32)
        self.P1_device = cl_array.to_device(device_queue, P1)

        self.built = True

    @check_built
    def predict(self, z, output_only=True, pre_activation_of_input=None):
        out = cl_array.empty_like(z)
        PolynomialTransformationInterface.polynomial_transform_1d(self.P1_device, z, self.M1,
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
        PolynomialTransformationInterface.polynomial_transform_1d(self.P1_device.T, delta, self.M1,
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
        return 'Kravchuk Transformation', f'Output Shape {self.input_shape}'
