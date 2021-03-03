from PySoap2 import get_activation_function
from PySoap2.validation import check_layer
from PySoap2.layers import Layer

import numpy as np


class ElementWise(Layer):
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

        l1_ratio : float
            The l1 regularisation strength on the weights

        built : bool
            Has the model been initialised

        Notes
        -----
        Since only element wise multiplication/addition is performed, the input
        shape is the same as the output shape, which is also the same shape as the bias
        and weight
    """

    def __init__(self, activation_function, l1_ratio=0.0, activation_kwargs=None):
        """ Initialise class

            Parameters
            ----------
            activation_function : str
                The name of the activation function of this layer
        """
        self.activation_function = activation_function
        self.activation_kwargs = {} if activation_kwargs is None else activation_kwargs

        self.l1_ratio = l1_ratio

        self.input_shape = None
        self.output_shape = None
        self.W = None
        self.b = None

        self.built = False

    def build(self, input_shape):
        """ Initialises the weight and bias units

            Parameters
            ----------
            input_shape : tuple of int
                The output shape of the previous layer.
        """

        self.input_shape = input_shape
        self.output_shape = input_shape

        self.W = np.random.rand(*self.input_shape) - 0.5
        self.b = np.random.rand(*self.input_shape) - 0.5

        self.built = True

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

        a = self.W[None, ...] * z + self.b[None, ...]
        if output_only:
            return self.activation_function_(a)
        return a, self.activation_function_(a)

    def get_delta_backprop_(self, g_prime, new_delta, *args):
        """ Returns the delta for the previous layer, delta^{k-1}_{m,j}.

            Parameters
            ----------
            g_prime : (N, ...) np.array
                Should be the derivative of the output of the previous layer, g'_{k-1}(a^{k-1}_{m,j})
            new_delta : (N, ...) np.array
                The delta for this layer, delta^k_{m, j}

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

        return new_delta * self.W[None, ...] * g_prime

    def get_weight_grad_(self, delta, prev_z):
        """ Returns the associated partial S/partial W^k, that is
            the gradient with respect to the weight matrix in the kth layer

            Parameters
            ----------
            delta : (N, ...) np.array
                Should be delta_k
            prev_z : (N, ...) np.array
                This should be the input of this layer (z_{k-1})

            Returns
            -------
            (...) np.array, (...) np.array
                The first array is the gradient for the bias unit
                The second array is the gradient for the weight matrix
        """
        check_layer(self)

        # weight_updates = np.add.reduce(delta * prev_z, axis = 0)
        # bias_updates = np.add.reduce(delta, axis = 0)
        # weight_updates = np.sum(delta * prev_z, axis = 0)
        weight_grad = np.einsum('i...,i...', delta, prev_z)
        bias_grad = np.sum(delta, axis=0)

        return bias_grad, weight_grad

    def update_parameters_(self, bias_updates, weight_updates):
        """ Perform an update to the weights by descending down the gradient

            Parameters
            ----------
            bias_updates : np.array (of dimension k)
                The gradients for the bias units
            weight_updates : np.array (of dimension k)
                The gradients for the weight matrix
        """
        check_layer(self)

        self.W -= weight_updates
        self.b -= bias_updates

    def get_weights(self):
        check_layer(self)
        return self.W, self.b

    def summary_(self):
        check_layer(self)
        return f'Element Wise', f'Output Shape {(None, *self.output_shape)}'

    @property
    def activation_function_(self):
        return get_activation_function(self.activation_function, **self.activation_kwargs)

    def __str__(self):
        return f'Element Wise: Output Shape {(None, *self.output_shape)}'

    def __call__(self, layer):
        self.parent = (layer,)
        return self
