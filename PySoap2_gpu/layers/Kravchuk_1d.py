import numpy as np

import pyopencl as cl
import pyopencl.array as cl_array

from PySoap2.layers.PolynomialTransformations import Kravchuk

from PySoap2_gpu.layers import Polynomial_1d

from .ValueChecks import check_built


class Kravchuk_1d(Polynomial_1d):
    """ 1-d Kravchuk transform the input to this layer

        Notes
        -----
        The input to this layer is assumed to either be 1 dimensional, or 2 dimensional data.
    """
    def __init__(self, p=0.5, inverse=False):
        Polynomial_1d.__init__(self, None)

        self.p = p
        self.inverse = inverse

    def build(self, device_context, device_queue):
        super().build(device_context, device_queue)

        P1 = Kravchuk.polynomials(self.M1 - 1, self.p).astype(np.float64)
        if self.inverse:
            P1 = P1.T

        self.P1 = cl_array.to_device(device_queue, P1)

        self.built = True

    @check_built
    def summary_(self):
        return 'Kravchuk Transformation', f'Output Shape {self.input_shape}'
