import numpy as np

from PySoap2 import get_activation_function
from PySoap2.layers import Layer
from PySoap2.validation import check_layer


class Dense(Layer):
    """ A fully connected layer

        Attributes
        ----------
        hidden_nodes : int
            The number of neurons in this layer
        g_name : str
            Name of the activation function
        built : bool
            Has the model been initialised
        output_shape : (k, ) tuple
            The shape of the output of this layer
        input_shape : (j, ) tuple
            The shape of the input of this layer
        W : (k, j) np.array
            The weight matrix
        b : (k, ) np.array
            The bias unit

        Notes
        -----
        It is assumed that the input to this layer is a flattened vector. As such, when passing
        a multidimensional input, use a `flatten` layer first
    """
    def __init__(self, hidden_nodes, activation_function, *arg, activation_kwargs=None, **kwargs):
        """ A fully connected layer

            Parameters
            ----------
            hidden_nodes : int
                The number of neurons in this layer
            activation_function : str
                The name of the activation function of this layer
            activation_kwargs : dict of str - :obj:, optional
                The keyword arguments for the activation function if it has hyper-parameters
        """
        self.hidden_nodes = hidden_nodes

        self.activation_function = activation_function
        self.activation_kwargs = {} if activation_kwargs is None else activation_kwargs

        self.output_shape = None
        self.input_shape = None
        self.W = None
        self.b = None

        self.basis = None
        self.coeffs = None

        self.built = False

    def build(self, input_shape):
        """ Initialises the weight and bias units

            Parameters
            ----------
            input_shape : 1 tuple of int
                The output shape of the previous layer. This will dictate the size of the weight matrix
        """
        self.output_shape = (self.hidden_nodes, )
        self.input_shape = input_shape

        # Initialise the the weight with Glorot-Uniform, a uniform distribution over [-limit, limit],
        # where limit = sqrt(6 / (fan_in + fan_out)) (fan_in is the number of input units in the weight
        # tensor and fan_out is the number of output units).
        limit = np.sqrt(6 / (np.prod(self.input_shape) + np.prod(self.output_shape)))
        self.W = np.random.uniform(low=-limit, high=limit, size=(*self.output_shape, *input_shape))
        self.b = np.zeros(self.output_shape)

        self.built = True

    def predict(self, z, output_only=True):
        """ Returns the output of this layer

            Parameters
            ----------
            z : (N, j) np.array
                z is assumed to be a list of all the inputs to be forward propagated. In particular
                it is assumed that the first index of z is the index that inputs is accessed by
            output_only : bool, optional
                If set to true, then this function will return only the prediction of the neural
                network. If set to false then this will return the outputs of the individual
                layers. Unless back propagation is being performed, this should be set to true.

            Returns
            -------
            (N, k) np.array
                The final output of the layer, post activation

            OR (if `output_only = False`)

            (N, k) np.array, (N, k) np.array
                The first np.array will store the output before it is passed through the activation
                function.
                The second np.array will store the output after it has passed through the
                activation function.
        """
        check_layer(self)

        out_a = z @ self.W.T + self.b

        if output_only:
            return self.activation_function_(out_a)
        return out_a, self.activation_function_(out_a)

    def get_delta_backprop_(self, g_prime, new_delta, *args):
        """ Returns the delta for the previous layer, delta^{k-1}_{m,j}.

            Notes
            -----
            We want to return delta^{k-1} because the `sequential` class does not have access to the
            weights, W. But it does know the values of g'_{k-1} and delta^k, due to forward propagation
            and the backwards nature of the back propagation algorithm.

            Parameters
            ----------
            g_prime : (N, j) np.array
                Should be the derivative of the ouput of the previous layer, g'_{k-1}(a^{k-1}_{m,j})
            new_delta : (N, k) np.array
                The delta for this layer, delta^k_{m, j}

            Returns
            -------
            np.array
                Returns delta of the previous layer, delta^{k-1}
        """
        check_layer(self)
        return g_prime*(new_delta @ self.W)

    def get_parameter_gradients_(self, delta, prev_z):
        """ Returns the associated partial S/partial W^k, that is
            the gradient with respect to the weight matrix in the kth layer

            Parameters
            ----------
            delta : (N, k) np.array
                In latex, this should be delta_k
            prev_z : (N, j) np.array
                This should be the output, post activation, of the previous layer (z_{k-1})

            Returns
            -------
            (N, k) np.array, (N, k) np.array
                The first array is the gradient for the bias unit
                The second array is the gradient for the weight matrix
        """
        check_layer(self)

        parameter_gradients = {'weight': delta.T @ prev_z, 'bias': np.sum(delta, axis=0)}

        return parameter_gradients

    def update_parameters_(self, parameter_updates):
        """ Perform an update to the weights by descending down the gradient

            Parameters
            ----------
            parameter_updates : dict of str - np.array
                The step size for the parameters as scheduled by the optimizer
        """
        check_layer(self)

        self.W -= parameter_updates['weight']
        self.b -= parameter_updates['bias']

    def get_weights(self):
        check_layer(self)
        return self.W, self.b

    def summary_(self):
        check_layer(self)
        return f'Dense {(self.hidden_nodes,)}', f'Output Shape {(None, *self.output_shape)}'

    @property
    def activation_function_(self):
        return get_activation_function(self.activation_function, **self.activation_kwargs)

    def __str__(self):
        return f'Dense: Output Shape {(None, *self.output_shape)}'

    def __call__(self, layer):
        self.parent = (layer,)
        return self
