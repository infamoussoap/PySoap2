from PySoap2.layers import Polynomial_2d

from .PolynomialTransformations import Hahn
from .LayerBuiltChecks import check_built


class Hahn_2d(Polynomial_2d):
    """ 2-d Hahn transform the input to this layer

        Notes
        -----
        The input to this layer is assumed to either be 2 dimensional, or 3 dimensional data.
    """
    def __init__(self, a=0, b=0, a2=None, b2=None):
        Polynomial_2d.__init__(self, None, None)

        self.a, self.b = a, b

        self.a2 = a if a2 is None else a2
        self.b2 = b if b2 is None else b2

    def build(self):
        """ Initialises the weight and bias units """
        super().build()

        input_shape = self.parents[0].output_shape

        n, m = input_shape[0], input_shape[1]
        self.P1 = Hahn.polynomials(n - 1, self.a, self.b)
        self.P2 = Hahn.polynomials(m - 1, self.a2, self.b2)

        self.built = True

    @check_built
    def summary_(self):
        return 'Hahn Transformation', f'Output Shape {self.input_shape}'
