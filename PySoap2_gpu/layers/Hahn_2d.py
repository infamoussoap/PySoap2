import numpy as np

import pyopencl as cl
import pyopencl.array as cl_array

from PySoap2.layers.PolynomialTransformations import Hahn

from PySoap2_gpu.layers import Polynomial_2d

from .ValueChecks import check_built


class Hahn_2d(Polynomial_2d):
    """ 2-d Hahn transform the input to this layer

        Notes
        -----
        The input to this layer is assumed to either be 2 dimensional, or 3 dimensional data.
    """

    def __init__(self, a=0, b=0, a2=None, b2=None, inverse=False):
        Polynomial_2d.__init__(self, None, None)

        self.a, self.b = a, b

        self.a2 = a if a2 is None else a2
        self.b2 = b if b2 is None else b2

        self.inverse = inverse

    def build(self, device_context, device_queue):
        super().build(device_context, device_queue)

        P1 = Hahn.polynomials(self.M1 - 1, self.a, self.b).astype(np.float32)
        P2 = Hahn.polynomials(self.M2 - 1, self.a, self.b).astype(np.float32)

        if self.inverse:
            P1 = P1.T
            P2 = P2.T

        self.P1 = cl_array.to_device(device_queue, P1)
        self.P2 = cl_array.to_device(device_queue, P2)

        self.built = True

    @check_built
    def summary_(self):
        return 'Hahn Transformation', f'Output Shape {self.input_shape}'
