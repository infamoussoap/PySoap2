import numpy as np

from PySoap2.layers import Layer
from PySoap2.layers.NetworkNode import NetworkNode
from PySoap2.layers.LayerBaseAttributes import LayerBaseAttributes

from .LayerBuiltChecks import check_built


class Flatten(NetworkNode, LayerBaseAttributes, Layer):
    """ Given a n-dimensional input, this layer will return the flatten representation
        of the input

        Attributes
        ----------
        input_shape : tuple
            The input shape
        output_shape : 1 tuple
            The output shape
        built : bool
            Has the layer been initialised

        Notes
        -----
        When a n-dimensional input is fed into a `Dense` layer, it needs to be flattened
        into a vector first. This `Flatten` class performs such flattening
    """

    def __init__(self):
        NetworkNode.__init__(self)
        LayerBaseAttributes.__init__(self)

    def build(self):
        """ Built/initialised the layer """
        input_shape = self.parents[0].output_shape

        self.input_shape = input_shape
        self.output_shape = (np.prod(input_shape),)

        self.built = True

    @check_built
    def predict(self, z, output_only=True, pre_activation_of_input=None):
        """ Returns the prediction of this layer

            Parameters
            ----------
            z : (N, *input_shape) np.array
                The input to be flattened
            output_only : bool, optional
                If set to true, then this function will return only the prediction of the neural
                network. If set to false then this will return the outputs of the individual
                layers. Unless back propagation is being performed, this should be set to true.
            pre_activation_of_input : (N, *input_shape) np.array
                The input, z, before it passed through the activation function

            Returns
            -------
            (N, *output_shape) np.array
                The flattened representation of the input

            OR (if `output_only = False`)

            (N, *input_shape) np.array, (N, *output_shape) np.array
                The first np.array will store the output before it has been reshaped
                The second np.array will store the output after it has been reshaped

            Notes
            -----
            Since this layer has no activation function,
        """

        if output_only:
            return z.reshape(len(z), self.output_shape[0])
        return pre_activation_of_input, z.reshape(len(z), self.output_shape[0])

    @check_built
    def get_delta_backprop_(self, g_prime, new_delta, *args, **kwargs):
        """ Returns the delta for the previous layer, delta^{k-1}_{m,j}.

            Parameters
            ----------
            g_prime : (N, *input_shape) np.array
                Should be the derivative of the ouput of the previous layer, g'_{k-1}(a^{k-1}_{m,j})
            new_delta : (N, *output_shape) np.array
                The delta for this layer, delta^k_{m, j}

            Returns
            -------
            (N, *input_shape) np.array

            Notes
            -----
            Since this is a pass through layer (i.e. linear activation), g_prime = 1, and so can be ignored.
            The key to this layer is that the delta of the k+1 layer needs to be reshaped
            for the k-1 layer
        """

        return new_delta.reshape(len(new_delta), *self.input_shape)

    @check_built
    def get_parameter_gradients_(self, *args, **kwargs):
        """ Returns the associated partial S/partial W^k, that is
            the gradient with respect to the weight matrix in the kth layer

            Returns
            -------
            dict

            Notes
            -----
            Since nothing in this layer is trainiable, the gradients is simply None
        """

        parameter_gradients = {}
        return parameter_gradients

    @check_built
    def update_parameters_(self, parameter_gradients):
        """ Perform an update to the weights by descending down the gradient

            Notes
            -----
            Since nothing in this layer is trainiable, we can simply pass
        """
        pass

    @check_built
    def get_weights(self):
        return None, None

    @check_built
    def summary_(self):
        return f'Flatten', f'Output Shape {(None, *self.output_shape)}'

    def __str__(self):
        return f'Flatten'

    @property
    def activation_function_(self):
        def reshaped_activation_function(x, grad=False):
            parent = self.parents[0]
            post_activation = parent.activation_function_(x, grad=grad)

            if parent.activation_function == 'linear' and grad:
                return post_activation
            
            return self.predict(post_activation, output_only=True)

        return reshaped_activation_function
