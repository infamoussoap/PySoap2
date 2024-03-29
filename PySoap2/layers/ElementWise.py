import numpy as np
from functools import reduce

from PySoap2.layers import Layer
from PySoap2.layers.NetworkNode import NetworkNode
from PySoap2.layers.LayerBaseAttributes import LayerBaseAttributes

from .LayerBuiltChecks import check_built


class ElementWise(NetworkNode, LayerBaseAttributes, Layer):
    """ A ElementWise layer (previously named push forward) - Where forward propagation is defined as
        element wise multiplication of the weight with the input, and element wise addition of the bias

        Attributes
        ----------
        activation_function : str
            The activation function
        activation_kwargs : dict
            Keyword arguments for the activation function, if there are any

        output_shape : k tuple
            The shape of the output of this layer
        input_shape : k tuple
            The shape of the input of this layer
        W : np.array (of dimension k)
            The weight matrix
        b : np.array (of dimension k)
            The bias unit

        built : bool
            Has the model been initialised

        Notes
        -----
        Since only element wise multiplication/addition is performed, the input
        shape is the same as the output shape, which is also the same shape as the bias
        and weight
    """

    def __init__(self, activation_function, activation_kwargs=None, weight_decay=0.0):
        """ Initialise class

            Parameters
            ----------
            activation_function : str
                The name of the activation function of this layer
        """
        NetworkNode.__init__(self)
        LayerBaseAttributes.__init__(self)

        self.activation_function = activation_function
        self.activation_kwargs = {} if activation_kwargs is None else activation_kwargs

        self.weight_decay = weight_decay

        self.W = None
        self.b = None

    def build(self):
        """ Initialises the weight and bias units """
        input_shape = self.parents[0].output_shape

        self.input_shape = input_shape
        self.output_shape = input_shape

        self.W = np.random.rand(*self.input_shape) - 0.5
        self.b = np.random.rand(*self.input_shape) - 0.5

        self.built = True

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

        a = self.W[None, ...] * z + self.b[None, ...]
        if output_only:
            return self.activation_function_(a)
        return a, self.activation_function_(a)

    @check_built
    def get_delta_backprop_(self, g_prime, new_delta, *args):
        """ Returns the delta for the previous layer, delta^{k-1}_{m,j}.

            Parameters
            ----------
            g_prime : (N, ...) np.array
                Should be the derivative of the output of the previous layer, g'_{k-1}(a^{k-1}_{m,j})
            new_delta : list of (N, ...) np.array

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
        delta = reduce(sum, new_delta)

        return delta * self.W[None, ...] * g_prime

    @check_built
    def get_parameter_gradients_(self, delta, prev_z, e=1e-7):
        """ Returns the associated partial S/partial W^k, that is
            the gradient with respect to the weight matrix in the kth layer

            Parameters
            ----------
            delta : (N, ...) np.array
                Should be delta_k
            prev_z : (N, ...) np.array
                This should be the input of this layer (z_{k-1})
            e : float, optional
                Cut off for machine precision 0

            Returns
            -------
            dict of str - np.array
                Keys are the parameters for the softchop function, with the corresponding values their
                gradients
        """
        delta = reduce(sum, delta)

        if abs(self.weight_decay) > e:
            parameter_gradients = {'weight': np.einsum('i...,i...', delta, prev_z) + self.weight_decay * self.W,
                                   'bias': np.sum(delta, axis=0)}
        else:
            parameter_gradients = {'weight': np.einsum('i...,i...', delta, prev_z), 'bias': np.sum(delta, axis=0)}

        return parameter_gradients

    @check_built
    def update_parameters_(self, parameter_updates):
        """ Perform an update to the weights by descending down the gradient

            Parameters
            ----------
            parameter_updates : dict of str - np.array
                The step size for the parameters as scheduled by the optimizer
        """

        self.W -= parameter_updates['weight']
        self.b -= parameter_updates['bias']

    @check_built
    def get_weights(self, as_dict=False):
        if as_dict:
            return {'W': self.W, 'b': self.b}
        return self.W, self.b

    @check_built
    def summary_(self):
        return f'Element Wise', f'Output Shape {(None, *self.output_shape)}'

    def __str__(self):
        return f'Element Wise: Output Shape {(None, *self.output_shape)}'
