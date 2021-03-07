import numpy as np

from functools import partial

from PySoap2.layers import Layer
from PySoap2.layers.NetworkNode import NetworkNode
from PySoap2.layers.LayerBaseAttributes import LayerBaseAttributes
from PySoap2.validation import check_layer


class MultiSoftChop:
    """ The element wise softchop function, where each element gets their own hyper-parameters
        This class implements the evalaution of the softchop function, as well as the partial
        derivatives with respect to
            - the input
            - the a1, a2, epsilon1, epsilon2 hyper-parameters


        Notes
        -----
        The softchop function has 4 hyper-parameters:
            a1 : np.array (of dimension k)
                Threshold of positive x values
            a2 : np.array (of dimension k)
                Threshold of positive x values
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
        Training on this layer happens on the individual hyer-parameters of the softchop
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
            Threshold of positive x values
        epsilon1 : np.array (of dimension k)
            Gradient of acceptance for positive x values
        epsilon2 : np.array (of dimension k)
            Gradient of acceptance for negative x values

        built : bool
            Has the layer been built
    """

    def __init__(self):
        NetworkNode.__init__(self)
        LayerBaseAttributes.__init__(self)

        self.activation_function = 'multi_softchop'

        # Initialising Attributes of Class
        self.a1 = None
        self.a2 = None
        self.epsilon1 = None
        self.epsilon2 = None

    def build(self):
        """ Initialise the Softchop Hyper-parameters """
        input_shape = self.parents[0].output_shape

        self.input_shape = input_shape
        self.output_shape = input_shape

        self.a1 = np.random.rand(*self.input_shape) * 2
        self.a2 = np.random.rand(*self.input_shape) * 2

        self.epsilon1 = np.random.rand(*self.input_shape)
        self.epsilon2 = np.random.rand(*self.input_shape)

        self.clip_parameters()

        self.built = True

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

            Recommend mimimum values for epsilon1, epsilon2 is 0.001. Anything lower
            produces no meaningful change. Caution for epsilon too small, as it will
            be inverterd for some of the calculations.
            a1 and a2 can be anything, 0.001 was choosen arbitrarily for this value
        """

        self.a1 = np.clip(self.a1, min_a, None)
        self.a2 = np.clip(self.a2, min_a, None)

        self.epsilon1 = np.clip(self.epsilon1, min_epsilon, None)
        self.epsilon2 = np.clip(self.epsilon2, min_epsilon, None)

    def predict(self, z, output_only=True):
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
        check_layer(self)

        a = self.activation_function_(z)
        if output_only:
            return a
        return a, a

    def get_delta_backprop_(self, g_prime, new_delta, prev_z):
        """ Returns the delta for the previous layer, delta^{k-1}_{m,j}.

            Parameters
            ----------
            g_prime : (N, ...) np.array
                Should be the derivative of the ouput of the previous layer, g'_{k-1}(a^{k-1}_{m,j})
            new_delta : (N, ...) np.array
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
        check_layer(self)

        return new_delta * g_prime * self.activation_function_(prev_z, grad=True)

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
            (2, ...) np.array, (2, ...) np.array
                The first array is the gradient for the `a` hyper-parameters. Note that
                the arrays start with dimension 2 - the 0th entry corresponds to the `a1` gradients
                while the 1st entry corresponds to the `a2` gradients

                The second array is the gradient for the `epsilon` hyper-parameters. Similarly to
                the `a` hyper-parameters, this also starts with dimension 2 - the 0th entry for the
                `epsilon1` gradients and the 1st entry for the `epsilon2` gradients
        """
        check_layer(self)

        # weight_updates = np.add.reduce(delta * prev_z, axis = 0)
        # bias_updates = np.add.reduce(delta, axis = 0)
        # weight_updates = np.sum(delta * prev_z, axis = 0)

        kwargs = {'x': prev_z, 'a1': self.a1, 'a2': self.a2, 'epsilon1': self.epsilon1, 'epsilon2': self.epsilon2}

        parameter_gradients = {'a1': np.einsum('i...,i...', delta, MultiSoftChop.da1(**kwargs)),
                               'a2': np.einsum('i...,i...', delta, MultiSoftChop.da2(**kwargs)),
                               'epsilon1': np.einsum('i...,i...', delta, MultiSoftChop.de1(**kwargs)),
                               'epsilon2': np.einsum('i...,i...', delta, MultiSoftChop.de2(**kwargs))}

        return parameter_gradients

    def update_parameters_(self, parameter_updates):
        """ Update the softchop hyper-parameters by descending down the gradient

            Parameters
            ----------
            parameter_updates : dict of str - np.array
                The step size for the parameters as scheduled by the optimizer
        """
        check_layer(self)

        self.a1 -= parameter_updates['a1']
        self.a2 -= parameter_updates['a2']

        self.epsilon1 -= parameter_updates['epsilon1']
        self.epsilon2 -= parameter_updates['epsilon2']

        self.clip_parameters()

    def get_weights(self):
        check_layer(self)

        return np.array([self.a1, self.a2]), np.array([self.epsilon1, self.epsilon2])

    def summary_(self):
        check_layer(self)

        return f'SoftChop', f'Output Shape {(None, *self.output_shape)}'

    @property
    def activation_function_(self):
        return partial(MultiSoftChop.eval, a1=self.a1, a2=self.a2,
                       epsilon1=self.epsilon1, epsilon2=self.epsilon2)

    def __str__(self):
        return f'SoftChop: Output Shape {(None, *self.output_shape)}'
