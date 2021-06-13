import numpy as np
from functools import reduce

import pyopencl as cl
import pyopencl.array as cl_array

from PySoap2_gpu.layers import Layer
from PySoap2_gpu.layers.NetworkNode import NetworkNode
from PySoap2_gpu.layers.LayerBaseAttributes import LayerBaseAttributes

from .PolynomialTransformation import PolynomialTransformationInterface

from .ValueChecks import check_built


class Polynomial_2d(NetworkNode, LayerBaseAttributes, Layer):
    """ 2-d Kravchuk transform the input to this layer

        Notes
        -----
        The input to this layer is assumed to either be 2 dimensional, or 3 dimensional data.
    """
    def __init__(self, P1, P2=None):
        NetworkNode.__init__(self)
        LayerBaseAttributes.__init__(self)

        self.P1 = P1
        self.P2 = P2
        self.M1 = None
        self.M2 = None
        self.M3 = None

    def build(self, device_context, device_queue):
        """ Initialises the weight and bias units """
        self.device_context = device_context
        self.device_queue = device_queue

        if not PolynomialTransformationInterface.initialize:
            PolynomialTransformationInterface(device_context, device_queue)

        input_shape = self.parents[0].output_shape

        if len(input_shape) != 2 and len(input_shape) != 3:
            raise ValueError('Input to Kravchuk_2d assumed to either be 2d or 3d data, not '
                             f'{len(input_shape)}d')

        self.input_shape = input_shape
        self.output_shape = input_shape

        self.M1 = np.int32(input_shape[0])
        self.M2 = np.int32(input_shape[1])
        self.M3 = None if len(input_shape) == 2 else np.int32(input_shape[2])

        self.built = True

    @check_built
    def predict(self, z, output_only=True, pre_activation_of_input=None):
        out = cl_array.empty_like(z)
        PolynomialTransformationInterface.polynomial_transform_2d(self.P1, self.P2, z, self.M1,
                                                                  self.M2, self.M3,
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
        PolynomialTransformationInterface.polynomial_transform_2d(self.P1.T, self.P2.T, delta,
                                                                  self.M1, self.M2, self.M3,
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
