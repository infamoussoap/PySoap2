import numpy as np

import pyopencl as cl
import pyopencl.array as cl_array

from PySoap2.layers.PolynomialTransformations import Kravchuk

from PySoap2_gpu.layers import Polynomial_2d

from .ValueChecks import check_built


class Kravchuk_2d(Polynomial_2d):
    """ 2-d Kravchuk transform the input to this layer

        Notes
        -----
        The input to this layer is assumed to either be 2 dimensional, or 3 dimensional data.
    """
    def __init__(self, p=0.5, p2=None, inverse=False):
        Polynomial_2d.__init__(self, None, None)

        self.p = p
        self.p2 = p if p2 is None else p2
        self.inverse = inverse

    def build(self, device_context, device_queue):
        super().build(device_context, device_queue)

        P1 = Kravchuk.polynomials(self.M1 - 1, self.p).astype(np.float64)
        P2 = Kravchuk.polynomials(self.M2 - 1, self.p2).astype(np.float64)

        if self.inverse:
            P1 = P1.T
            P2 = P2.T

        self.P1 = cl_array.to_device(device_queue, P1)
        self.P2 = cl_array.to_device(device_queue, P2)

        self.built = True

    @check_built
    def summary_(self):
        return 'Kravchuk Transformation', f'Output Shape {self.input_shape}'
