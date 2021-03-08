from PySoap2.layers import Layer
from PySoap2.layers.NetworkNode import NetworkNode
from PySoap2.layers.LayerBaseAttributes import LayerBaseAttributes

from .LayerBuiltChecks import check_built


class Input(NetworkNode, LayerBaseAttributes, Layer):
    """ This class will determine the input dimensions of the dataset

        Notes
        -----
        This class must always be the first `sequential_layer` that is added to an instance
        of the `sequential` class. This sequential layer will be given the shape of the input
        and will determine the shape of later `sequential_layer` in the network
        Also, note that the input is essentially a pass through layer, that is not trainable

        Attributes
        ----------
        input_shape : tuple of int
            The input shape of a datapoint
        output_shape : tuple of int
            The output shape of this layer, which is equal to the `input_shape`
        built : bool
            Has this layer been initialised
    """

    def __init__(self, input_shape):
        """ Creates a `Input` class with a given input shape

            Parameters
            ----------
            input_shape : tuple of int
                The shape of a given data point
        """
        NetworkNode.__init__(self)
        LayerBaseAttributes.__init__(self)

        self.input_shape = (*input_shape, )
        self.output_shape = (*input_shape, )

        # This layer is assumed to be built from creation
        self.built = True

    def build(self):
        """ Initialises the layer

            Notes
            -----
            Since this is simply a pass through layer, there is no initialization needed.
            This method is only written so as to make the `Layer` uniform in
            implementation
        """
        pass

    @check_built
    def predict(self, z, *args, output_only=True, **kwargs):
        """ Returns the output of this layer

            Parameters
            ----------
            z : np.array
                z is assumed to be a list of all the inputs to be forward propagated. In particular
                it is assumed that the first index of z is the index that inputs is accessed by
            output_only : :obj:`bool`, optional
                If set to true, then this function will return only the prediction of the neural
                network. If set to false then this will return the outputs of the individual
                layers. Unless back propagation is being performed, this should be set to true.

            Returns
            -------
            np.array
                The final output of the layer

            OR (if `output_only = False`)

            np.array, np.array
                The first np.array will store the output before it is passed through the activation
                function.
                The second np.array will store the output after it has passed through the
                activation function.
        """

        if output_only:
            return z
        else:
            return z, z

    @check_built
    def get_delta_backprop_(self, g_prime, new_delta, *args):
        """ Since this layer has no parameters, there is no implementation needed
        """

        return None

    @check_built
    def get_parameter_gradients_(self, delta, prev_z):
        """ Since this layer has no parameters, there is no implementation needed """

        parameter_gradients = {}
        return parameter_gradients

    @check_built
    def update_parameters_(self, parameter_updates):
        """ Since this layer has no parameters, there is no implementation needed """

        pass

    @check_built
    def get_weights(self):
        return None, None

    @check_built
    def summary_(self):
        return f'Input', f'Input Shape  {(None, *self.input_shape)}'

    def __str__(self):
        return f'Input: Input Shape  {(None, *self.input_shape)}'
