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


class Kravchuk_2d(NetworkNode, LayerBaseAttributes, Layer):
    """ 2-d Kravchuk transform the input to this layer

        Notes
        -----
        The input to this layer is assumed to either be 2 dimensional, or 3 dimensional data.
    """
    def __init__(self, p=0.5, p2=None):
        NetworkNode.__init__(self)
        LayerBaseAttributes.__init__(self)

        self.p = p
        self.p2 = p if p2 is None else p2

        self.P1_device = None
        self.P2_device = None
        self.M1_device = None
        self.M2_device = None
        self.M3_device = None

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

        m1 = input_shape[0]
        m2 = input_shape[1]
        m3 = None if len(input_shape) == 2 else input_shape[2]
        P1 = kravchuk_polynomials(m1 - 1, self.p).astype(np.float32)
        P2 = kravchuk_polynomials(m2 - 1, self.p2).astype(np.float32)

        self.P1_device = cl_array.to_device(device_queue, P1)
        self.P2_device = cl_array.to_device(device_queue, P2)
        self.M1_device = cl_array.to_device(device_queue, np.array(m1, dtype=np.int32))
        self.M2_device = cl_array.to_device(device_queue, np.array(m2, dtype=np.int32))
        if m3 is not None:
            self.M3_device = cl_array.to_device(device_queue, np.array(m3, dtype=np.int32))

        self.built = True

    @check_built
    def predict(self, z, output_only=True, pre_activation_of_input=None):
        out = cl_array.empty_like(z)
        PolynomialTransformationInterface.polynomial_transform_2d(self.P1_device, self.P2_device, z, self.M1_device,
                                                                  self.M2_device, self.M3_device,
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
        PolynomialTransformationInterface.polynomial_transform_2d(self.P1_device.T, self.P2_device.T, delta,
                                                                  self.M1_device, self.M2_device, self.M3_device,
                                                                  self.input_length_device, out)

        return out

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
