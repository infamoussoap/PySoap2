from PySoap2.layers import Polynomial_1d
from .PolynomialTransformations import Hahn

from .LayerBuiltChecks import check_built


class Hahn_1d(Polynomial_1d):
    """ 1-d Hahn transform the input to this layer

        Notes
        -----
        The input to this layer is assumed to either be 1 dimensional, or 2 dimensional data.
    """
    def __init__(self, a=0, b=0, inverse=False):
        Polynomial_1d.__init__(self, None)

        self.a = a
        self.b = b

        self.inverse = inverse

    def build(self):
        """ Initialises the weight and bias units """
        super().build()

        input_shape = self.parents[0].output_shape

        n = input_shape[0]
        self.P1 = Hahn.polynomials(n - 1, self.a, self.b)

        if self.inverse:
            self.P1 = self.P1.T

        self.built = True

    @check_built
    def summary_(self):
        return 'Hahn Transformation', f'Output Shape {self.input_shape}'
