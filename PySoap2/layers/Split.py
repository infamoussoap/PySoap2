import numpy as np

from PySoap2.validation import check_layer
from PySoap2.layers import Layer
from PySoap2.layers.NetworkNode import NetworkNode
from PySoap2 import get_activation_function


class SplitChild(NetworkNode, Layer):
    """ This class is where the splitting is performed. The input (or parent to split node)
        is passed into Split which is then passed to SplitChild to perform the split

        Attributes
        ----------
        mask : np.array
            Assumed to be a np.array with bool entries. Entries where the mask is true
            are the positions of the input that will be returned

        input_shape : tuple of int
            The input shape of this layer
        output_shape : (i, )
            A 1-tuple - The output is simple the number of positions
        activation_function : str
            The name of the activation function. Note that since this layer doesn't do anything,
            this attribute is set to linear

        built : bool
            Has the model been built
    """

    def __init__(self, mask):
        """ Initialise the layer by passing the mask and the parent Split Node

            Parameters
            ----------
            mask : np.array (of bool)
                The mask for this instance
        """
        NetworkNode.__init__(self)

        self.mask = mask.astype(bool)

        self.activation_function = 'linear'

        self.input_shape = None
        self.output_shape = None

        self.built = False

    def build(self):
        """ Build the layer by determining the input and output shape """
        input_shape = self.parents[0].output_shape

        self.input_shape = input_shape
        self.output_shape = (np.sum(self.mask),)

        self.built = True

    def predict(self, z, output_only=True):
        """ Forward propagate the splitting

            Parameters
            ----------
            z : (N, *input_shape) np.array
                The input for this layer
            output_only : bool, optional
                If set to true, then only the split will be returned
                Otherwise, the original input and the split will be returned

            Returns
            -------
            (N, i) np.array
                The split if output_only=True

            OR

            (N, *input_shape) np.array, (N, i) np.array
                If output_only=False, the first position will be the original input
                with the second being the split of the input
        """
        check_layer(self)

        if output_only:
            return z[:, self.mask]
        return z, z[:, self.mask]

    def get_delta_backprop_(self, g_prime, new_delta, *args, **kwargs):
        """ Returns delta^{k-1}

            Notes
            -----
            This layer essential performs a mapping from the input into a subset
            of the input. As such, to find delta, we essentially need to do the reverse
            mapping. But this should be done in the parent Split node, not here.

            Parameters
            ----------
            g_prime : (N, *input_shape) np.array
                Should be g'_{k-1}
            new_delta : (N, *output_shape) np.array
                Should be delta^k

            Returns
            -------
            (N, *output_shape)
        """

        check_layer(self)
        return new_delta

    def get_parameter_gradients_(self, new_delta, prev_z):
        """ This method returns the gradients of the parameters for this layer. But
            since this layer has no trainable parameters, it has no gradients

            Notes
            -----
            This method is independent of the arguments, when calling this layer
            the argument should be np.array

            Parameters
            ----------
            new_delta : (N, *output_shape) np.array
                The delta for the k-th layer, should be delta^k
            prev_z : (N, *input_shape) np.array
                The input of this layer, should be z^{k-1}

            Returns
            -------
            None, None
        """

        parameter_gradients = {}
        return parameter_gradients

    def update_parameters_(self, parameter_updates):
        """ This layer has no trainiable parameters so nothing will be performed

            Notes
            -----
            Because this layer has no traininable parameters, the arguments passed
            into this method should be :obj:None, instead of np.array

            Parameters
            ----------
            parameter_updates : dict of str - np.array
        """
        pass

    def get_weights(self):
        """ This layer has no trainiable parameters, so error will be raised

            Raises
            ------
            AttributeError
        """
        raise AttributeError("Split Layer has no weights/parameters.")

    def summary_(self):
        check_layer(self)
        return 'SplitChild Layer', f'Output Shape {(None, *self.output_shape)}'

    def activation_function_(self):
        return get_activation_function(self.activation_function)


class Split(NetworkNode, Layer):
    """ Breaks up the input into two seperate outputs, such that when the outputs
        are combined, it will be equal to the original input. Note that because
        the break can be highly irregular, the outputs will be flatten arrays

        Notes
        -----
        This Split layer effectivly does nothing. All it does is provide an interface
        between the input (parent of the Split) to the split outputs

        Attributes
        ----------
        mask : np.array
            The mask will determine how to split the input. In particular
            the input at the mask will be the left child (left output), and
            the rest will be the right child (right output)

        input_shape : tuple of int
            The input shape of this layer
        output_shape : (i, )
            A 1-tuple - The output is simple the number of positions
        activation_function : str
            The name of the activation function. Note that since this layer doesn't do anything,
            this attribute is set to linear

        built : bool
            Has the model been built
    """

    def __init__(self, mask):
        """ Initialise the Split layer by giving the mask

            Notes
            -----
            The mask refers to the positions that will be returned from the left child.
            Everything else (i.e. ~mask) will be returned from the right child

            Parameters
            ----------
            mask : np.array
                This is not assumed to be bool, and will be converted to bool types
                on runtime.
        """
        NetworkNode.__init__(self)

        self.mask = mask.astype(bool)

        self.activation_function = 'linear'

        self.input_shape = None
        self.output_shape = None

        self.built = False

    def build(self):
        """ Initilise the layer

            Notes
            -----
            The output_shape is the same as the input_shape because the input must
            be based onto the children for it to be split
        """
        input_shape = self.parents[0].output_shape

        self.input_shape = input_shape
        self.output_shape = input_shape

        self.built = True

    def predict(self, z, output_only=True):
        """ Returns the prediction of this layer

            Notes
            -----
            Because this layer is only an interface between the parent Split node
            to the SplitChild nodes, this effectively does nothing
        """
        if output_only:
            return z
        return z, z

    def get_delta_backprop_(self, g_prime1, new_delta1, g_prime2, new_delta2, *args, **kwargs):
        """ Returns delta^{k-1}

            Notes
            -----
            Since the Split layer maps the input into 2 outputs, all we need to do to find
            delta^{k-1} is perform the reverse map of the 2 outputs
            Also note that g_prime1 and g_prime2 should be the same np.array, as this defines
            g'_{k-1}, that is the input for this layer

            Parameters
            ----------
            g_prime1 : (N, *input_shape) np.array
                Should be g'_{k-1} for the left child
            new_delta1 : (N, *left_output_shape) np.array
                Should be delta^k for the left child
            g_prime2 : (N, *input_shape) np.array
                Should be g'_{k-1} for the right child
            new_delta2 : (N, *right_output_shape) np.array
                Should be delta^k for the right child

            Returns
            -------
            (N, *input_shape) np.array
        """
        check_layer(self)

        out_delta = np.zeros(len(new_delta1), *self.input_shape)

        out_delta[:, self.mask] = g_prime1
        out_delta[:, ~self.mask] = g_prime2

        return out_delta

    def get_parameter_gradients_(self, new_delta, prev_z):
        """ This method returns the gradients of the parameters for this layer. But
            since this layer has no trainable parameters, it has no gradients

            Notes
            -----
            While this method is independent of the arguments, when calling this layer
            the argument should be np.array

            Parameters
            ----------
            new_delta : (N, *output_shape) np.array
                The delta for the k-th layer, should be delta^k
            prev_z : (N, *input_shape) np.array
                The input of this layer, should be z^{k-1}

            Returns
            -------
            None, None
        """
        return None, None

    def update_parameters_(self, bias_updates, weight_updates):
        """ This layer has no trainiable parameters so nothing will be performed

            Notes
            -----
            Because this layer has no traininable parameters, the arguments passed
            into this method should be :obj:None, instead of np.array

            Parameters
            ----------
            bias_updates : None
            weight_updates : None
        """
        pass

    def get_weights(self):
        """ This layer has no trainiable parameters, so error will be raised

            Raises
            ------
            AttributeError
        """
        raise AttributeError("Split Layer has no weights/parameters.")

    def summary_(self):
        check_layer(self)
        return 'Split Layer', f'Output Shape {(None, *self.output_shape)}'

    def activation_function_(self):
        return get_activation_function(self.activation_function)

    @property
    def left(self):
        """ This returns the left child node in the Split layer, which will return
            the input at the positions as dictated by the mask

            Returns
            -------
            :obj:SplitChild
        """
        left_child = SplitChild(self.mask)
        left_child(self)

        return left_child

    @property
    def right(self):
        """ This returns the right child node in the Split layer, which will return
            the input at the positions as dictated by conjugate mask (i.e. ~mask)

            Returns
            -------
            :obj:SplitChild
        """
        right_child = SplitChild(~self.mask)
        right_child(self)

        return right_child
