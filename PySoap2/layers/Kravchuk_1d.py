from PySoap2.layers import Polynomial_1d
from .PolynomialTransformations import Kravchuk

from .LayerBuiltChecks import check_built


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

    def build(self):
        """ Initialises the weight and bias units """
        super().build()

        input_shape = self.parents[0].output_shape

        n = input_shape[0]
        self.P1 = Kravchuk.polynomials(n - 1, self.p)
        if self.inverse:
            self.P1 = self.P1.T

        self.built = True

    @check_built
    def summary_(self):
        return 'Kravchuk Transformation', f'Output Shape {self.input_shape}'
