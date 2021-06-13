from PySoap2.layers import Polynomial_2d
from .PolynomialTransformations import Kravchuk
from .LayerBuiltChecks import check_built


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

    def build(self):
        """ Initialises the weight and bias units """
        super().build()

        input_shape = self.parents[0].output_shape

        n, m = input_shape[0], input_shape[1]
        self.P1 = Kravchuk.polynomials(n - 1, self.p)
        self.P2 = Kravchuk.polynomials(m - 1, self.p2)

        if self.inverse:
            self.P1, self.P2 = self.P1.T, self.P2.T

        self.built = True

    @check_built
    def summary_(self):
        return 'Kravchuk Transformation', f'Output Shape {self.input_shape}'
