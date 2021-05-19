import numpy as np
from functools import reduce

from PySoap2.layers import Layer
from PySoap2.layers.NetworkNode import NetworkNode
from PySoap2.layers.LayerBaseAttributes import LayerBaseAttributes

from .LayerBuiltChecks import check_built


class MultiSoftChop:
    """ The element wise softchop function, where each element gets their own hyper-parameters
        This class implements the evaluation of the softchop function, as well as the partial
        derivatives with respect to
            - the input
            - the a1, a2, epsilon1, epsilon2 hyper-parameters


        Notes
        -----
        The softchop function has 4 hyper-parameters:
            a1 : np.array (of dimension k)
                Threshold of positive x values
            a2 : np.array (of dimension k)
                Threshold of negative x values
            epsilon1 : np.array (of dimension k)
                Gradient of acceptance for positive x values
            epsilon2 : np.array (of dimension k)
                Gradient of acceptance for negative x values
    """

    @staticmethod
    def _positive_sigmoid(x):
        """ Stable sigmoid function for positive values """
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def _negative_sigmoid(x):
        """ Stable sigmoid function for negative values """
        exp = np.exp(x)
        return exp / (exp + 1)

    @staticmethod
    def _sigmoid(x):
        """ Stable sigmoid function for positive and negative values of x

            Parameters
            ----------
            x : np.array

            Returns
            -------
            np.array
                Same shape as x, the input
        """
        positive = x >= 0
        negative = ~positive

        result = np.zeros_like(x, dtype=np.float64)
        result[positive] = MultiSoftChop._positive_sigmoid(x[positive])
        result[negative] = MultiSoftChop._negative_sigmoid(x[negative])

        return result

    @staticmethod
    def eval(x, a1, a2, epsilon1, epsilon2, grad=False):
        """ Stable softchop function

            Parameters
            ----------
            x : np.array
            a1 : float or np.array
            a2 : float or np.array
            epsilon1 : float or np.array
            epsilon2 : float or np.array
            grad : bool, optional
                If true, the the derivative of the softchop function with respect to x will be returned.
                Otherwise evaluate the softchop function at the input x

            Returns
            -------
            np.array

            Notes
            -----
            This implementation of softchop relies on the assumption that the sigmoid function is stable.

            Also note that a1, a2, epsilon1, epsilon2 must be positive numbers. They won't make much
            sense if they are negative. Moreover, since we are dividing by epsilon, it is wise to not
            make them too close to 0.
        """
        if grad:
            return MultiSoftChop.dx(x, a1, a2, epsilon1, epsilon2)
        return x * (1 - (1 - MultiSoftChop._sigmoid((x - a1) / epsilon1)) * MultiSoftChop._sigmoid((x + a2) / epsilon2))

    @staticmethod
    def dx(x, a1, a2, epsilon1, epsilon2):
        """ Returns the partial derivative of softchop with respect to x """
        sig1 = MultiSoftChop._sigmoid((x - a1) / epsilon1)
        sig2 = MultiSoftChop._sigmoid((x + a2) / epsilon2)

        sig1_minus = MultiSoftChop._sigmoid(-(x - a1) / epsilon1)
        sig2_minus = MultiSoftChop._sigmoid(-(x + a2) / epsilon2)

        summand1 = 1 - (1 - sig1) * sig2
        summand2 = -sig1_minus * sig1 * sig2 / epsilon1
        summand3 = (1 - sig1) * sig2_minus * sig2 / epsilon2

        return summand1 - x * (summand2 + summand3)

    @staticmethod
    def da1(x, a1, a2, epsilon1, epsilon2):
        """ Returns the partial derivative of softchop with respect to a1 """
        sig1 = MultiSoftChop._sigmoid((x - a1) / epsilon1)
        sig2 = MultiSoftChop._sigmoid((x + a2) / epsilon2)

        sig1_minus = MultiSoftChop._sigmoid(-(x - a1) / epsilon1)

        return -x * sig1_minus * sig1 * sig2 / epsilon1

    @staticmethod
    def da2(x, a1, a2, epsilon1, epsilon2):
        """ Returns the partial derivative of softchop with respect to a2 """
        sig1 = MultiSoftChop._sigmoid((x - a1) / epsilon1)
        sig2 = MultiSoftChop._sigmoid((x + a2) / epsilon2)

        sig2_minus = MultiSoftChop._sigmoid(-(x + a2) / epsilon2)

        return -x * (1 - sig1) * sig2 * sig2_minus

    @staticmethod
    def de1(x, a1, a2, epsilon1, epsilon2):
        """ Returns the partial derivative of softchop with respect to epsilon1 """
        sig1 = MultiSoftChop._sigmoid((x - a1) / epsilon1)
        sig2 = MultiSoftChop._sigmoid((x + a2) / epsilon2)

        sig1_minus = MultiSoftChop._sigmoid(-(x - a1) / epsilon1)

        return -x * (x - a1) * sig1_minus * sig1 * sig2 / (epsilon1 ** 2)

    @staticmethod
    def de2(x, a1, a2, epsilon1, epsilon2):
        """ Returns the partial derivative of softchop with respect to epsilon2 """
        sig1 = MultiSoftChop._sigmoid((x - a1) / epsilon1)
        sig2 = MultiSoftChop._sigmoid((x + a2) / epsilon2)

        sig2_minus = MultiSoftChop._sigmoid(-(x + a2) / epsilon2)

        return x * (x + a2) * (1 - sig1) * sig2 * sig2_minus / (epsilon2 ** 2)


class SoftChop(NetworkNode, LayerBaseAttributes, Layer):
    """ A SoftChop layer where each input element has their own softchop function.
        Training on this layer happens on the individual hyper-parameters of the softchop
        function.

        Attributes
        ----------
        output_shape : k tuple
            The shape of the output of this layer
        input_shape : k tuple
            The shape of the input of this layer
        a1 : np.array (of dimension k)
            Threshold of positive x values
        a2 : np.array (of dimension k)
            Threshold of negative x values
        epsilon1 : np.array (of dimension k)
            Gradient of acceptance for positive x values
        epsilon2 : np.array (of dimension k)
            Gradient of acceptance for negative x values
        b : np.array (of dimension k)
            Bias unit

        include_bias : bool
            To include a bias unit inside the softchop function

        built : bool
            Has the layer been built
    """

    def __init__(self, include_bias=True):
        NetworkNode.__init__(self)
        LayerBaseAttributes.__init__(self)

        self.include_bias = include_bias

        # Initialising Attributes of Class
        self.a1 = None
        self.a2 = None
        self.epsilon1 = None
        self.epsilon2 = None

        self.b = None

    def build(self):
        """ Initialise the Softchop Hyper-parameters """
        input_shape = self.parents[0].output_shape

        self.input_shape = input_shape
        self.output_shape = input_shape

        self.a1 = np.random.rand(*self.input_shape) * 2
        self.a2 = np.random.rand(*self.input_shape) * 2

        self.epsilon1 = np.random.rand(*self.input_shape)
        self.epsilon2 = np.random.rand(*self.input_shape)

        self.b = (np.random.rand(*self.input_shape) - 0.5) if self.include_bias else np.zeros(self.input_shape)

        self.built = True

        self.clip_parameters()

    @check_built
    def clip_parameters(self, min_a=0.001, min_epsilon=0.001):
        """ Clip the hyper-parameters to be in a specific bound

            Parameters
            ----------
            min_a : float, optional
                The minimum value of a1 and a2
            min_epsilon : float, optional
                The minimum value of epsilon1 and epsilon2

            Notes
            -----
            The a1, a2, and epsilon1, epsilon2 hyper-parameters only make sense
            when they are positive

            Recommend minimum values for epsilon1, epsilon2 is 0.001. Anything lower
            produces no meaningful change. Caution for epsilon too small, as it will
            be inverted for some of the calculations.
            a1 and a2 can be anything, 0.001 was chosen arbitrarily for this value
        """

        self.a1 = np.clip(self.a1, min_a, None)
        self.a2 = np.clip(self.a2, min_a, None)

        self.epsilon1 = np.clip(self.epsilon1, min_epsilon, None)
        self.epsilon2 = np.clip(self.epsilon2, min_epsilon, None)

    @check_built
    def predict(self, z, output_only=True, **kwargs):
        """ Returns the output of this layer

            Parameters
            ----------
            z : (N, ...) np.array
                z is assumed to be a list of all the inputs to be forward propagated. In particular
                it is assumed that the first index of z is the index that inputs is accessed by
            output_only : bool, optional
                If set to true, then this function will return only the prediction of the neural
                network. If set to false then this will return the outputs of the individual
                layers. Unless back propagation is being performed, this should be set to true.

            Returns
            -------
            (N, ...) np.array
                The final output of the layer, post activation

            OR (if `output_only = False`)

            (N, ...) np.array, (N, ...) np.array
                The first np.array will store the output before it is passed through the activation
                function.
                The second np.array will store the output after it has passed through the
                activation function.
        """

        if self.include_bias:
            a = MultiSoftChop.eval(z + self.b, a1=self.a1, a2=self.a2, epsilon1=self.epsilon1, epsilon2=self.epsilon2)
        else:
            a = MultiSoftChop.eval(z, a1=self.a1, a2=self.a2, epsilon1=self.epsilon1, epsilon2=self.epsilon2)

        if output_only:
            return a
        return a, a

    @check_built
    def get_delta_backprop_(self, g_prime, new_delta, prev_z):
        """ Returns the delta for the previous layer, delta^{k-1}_{m,j}.

            Parameters
            ----------
            g_prime : (N, ...) np.array
                Should be the derivative of the output of the previous layer, g'_{k-1}(a^{k-1}_{m,j})
            new_delta : list of (N, ...) np.array
                The delta for this layer, delta^k_{m, j}
            prev_z : (N, ...) np.array
                The input for this layer, z^{n-1}

            Returns
            -------
            (N, ...) np.array
                Returns delta of the previous layer, delta^{k-1}

            Notes
            -----
            We want to return delta^{k-1} because the `sequential` class does not have access to the
            weights, W. But it does know the values of g'_{k-1} and delta^k, due to forward propagation
            and the backwards nature of the back propagation algorithm.
        """
        if self.include_bias:
            dz = MultiSoftChop.dx(prev_z + self.b, self.a1, self.a2, self.epsilon1, self.epsilon2)
        else:
            dz = MultiSoftChop.dx(prev_z, self.a1, self.a2, self.epsilon1, self.epsilon2)

        delta = np.sum(np.array(new_delta), axis=0)

        return delta * g_prime * dz

    @check_built
    def get_parameter_gradients_(self, delta, prev_z):
        """ Returns the associated partial S/partial W^k, that is
            the gradient with respect to the weight matrix in the kth layer

            Parameters
            ----------
            delta : (N, ...) np.array
                In latex, this should be delta_k
            prev_z : (N, ...) np.array
                This should be the output, post activation, of the previous layer (z_{k-1})

            Returns
            -------
            dict of str - np.array
                Keys are the parameters for the softchop function, with the corresponding values their
                gradients
        """

        if self.include_bias:
            kwargs = {'x': prev_z + self.b, 'a1': self.a1, 'a2': self.a2,
                      'epsilon1': self.epsilon1, 'epsilon2': self.epsilon2}
        else:
            kwargs = {'x': prev_z, 'a1': self.a1, 'a2': self.a2,
                      'epsilon1': self.epsilon1, 'epsilon2': self.epsilon2}

        delta = reduce(lambda x, y: x + y, delta)

        parameter_gradients = {'a1': np.einsum('i...,i...', delta, MultiSoftChop.da1(**kwargs)),
                               'a2': np.einsum('i...,i...', delta, MultiSoftChop.da2(**kwargs)),
                               'epsilon1': np.einsum('i...,i...', delta, MultiSoftChop.de1(**kwargs)),
                               'epsilon2': np.einsum('i...,i...', delta, MultiSoftChop.de2(**kwargs)),
                               'bias': np.einsum('i...,i...', delta, MultiSoftChop.dx(**kwargs))}

        return parameter_gradients

    @check_built
    def update_parameters_(self, parameter_updates):
        """ Update the softchop hyper-parameters by descending down the gradient

            Parameters
            ----------
            parameter_updates : dict of str - np.array
                The step size for the parameters as scheduled by the optimizer
        """

        self.a1 -= parameter_updates['a1']
        self.a2 -= parameter_updates['a2']

        self.epsilon1 -= parameter_updates['epsilon1']
        self.epsilon2 -= parameter_updates['epsilon2']

        if self.include_bias:
            self.b -= parameter_updates['bias']

        self.clip_parameters()

    @check_built
    def get_weights(self):
        return np.array([self.a1, self.a2]), np.array([self.epsilon1, self.epsilon2])

    @check_built
    def summary_(self):
        return f'SoftChop', f'Output Shape {(None, *self.output_shape)}'

    def __str__(self):
        return f'SoftChop: Output Shape {(None, *self.output_shape)}'
