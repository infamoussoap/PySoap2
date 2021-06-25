import numpy as np

from PySoap2.layers import Layer
from PySoap2.layers.NetworkNode import NetworkNode
from PySoap2.layers.LayerBaseAttributes import LayerBaseAttributes

from .LayerBuiltChecks import check_built


class SplitChild(NetworkNode, LayerBaseAttributes, Layer):
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
        LayerBaseAttributes.__init__(self)
        self.mask = mask.astype(bool)

    def build(self):
        """ Build the layer by determining the input and output shape """
        input_shape = self.parents[0].output_shape

        self.input_shape = input_shape
        self.output_shape = (np.sum(self.mask),)

        self.built = True

    @check_built
    def predict(self, z, output_only=True, pre_activation_of_input=None, **kwargs):
        """ Forward propagate the splitting

            Parameters
            ----------
            z : (N, *input_shape) np.array
                The input for this layer
            output_only : bool, optional
                If set to true, then only the split will be returned
                Otherwise, the original input and the split will be returned
            pre_activation_of_input : (N, *input_shape) np.array
                The input, z, before it passed through the activation function

            Returns
            -------
            (N, i) np.array
                The split if output_only=True

            OR

            (N, *input_shape) np.array, (N, i) np.array
                If output_only=False, the first position will be the original input
                with the second being the split of the input
        """

        if output_only:
            return z[:, self.mask]
        return pre_activation_of_input[:, self.mask], z[:, self.mask]

    @check_built
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
            new_delta : list of (N, *output_shape) np.array

            Returns
            -------
            (N, *output_shape)
        """

        return np.sum(np.array(new_delta), axis=0)

    @check_built
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
            {}
        """

        return {}

    @check_built
    def update_parameters_(self, parameter_updates):
        """ This layer has no trainable parameters so nothing will be performed

            Notes
            -----
            Because this layer has no trainable parameters, the arguments passed
            into this method should be :obj:None, instead of np.array

            Parameters
            ----------
            parameter_updates : dict of str - np.array
        """

        pass

    @check_built
    def get_weights(self, as_dict=False):
        if as_dict:
            return {}
        return None

    @check_built
    def summary_(self):

        return 'SplitChild Layer', f'Output Shape {(None, *self.output_shape)}'

    @property
    def activation_function_(self):
        return self.parents[0].activation_function_


class SplitLeftChild(SplitChild):
    """ There is no difference between the implementation of the Left and Right Child nodes.
        But these classes are created to make it clear which is the left and right node when
        looking at the repr of the instances
    """
    def __init__(self, mask):
        super().__init__(mask)


class SplitRightChild(SplitChild):
    """ There is no difference between the implementation of the Left and Right Child nodes.
            But these classes are created to make it clear which is the left and right node when
            looking at the repr of the instances
    """
    def __init__(self, mask):
        super().__init__(mask)


class Split(NetworkNode, LayerBaseAttributes, Layer):
    """ Breaks up the input into two separate outputs, such that when the outputs
        are combined, it will be equal to the original input. Note that because
        the break can be highly irregular, the outputs will be flatten arrays

        Notes
        -----
        This Split layer effectively does nothing. All it does is provide an interface
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
        LayerBaseAttributes.__init__(self)
        self.mask = mask.astype(bool)

        SplitLeftChild(self.mask)(self)
        SplitRightChild(~self.mask)(self)

    def build(self):
        """ Initialise the layer

            Notes
            -----
            The output_shape is the same as the input_shape because the input must
            be based onto the children for it to be split
        """
        input_shape = self.parents[0].output_shape

        self.input_shape = input_shape
        self.output_shape = input_shape

        self.built = True

    @check_built
    def predict(self, z, output_only=True, pre_activation_of_input=None):
        """ Returns the prediction of this layer

            Notes
            -----
            Because this layer is only an interface between the parent Split node
            to the SplitChild nodes, this effectively does nothing
        """
        if output_only:
            return z.reshape(-1, *self.output_shape)
        return pre_activation_of_input.reshape(-1, *self.input_shape), z.reshape(-1, *self.output_shape)

    @check_built
    def get_delta_backprop_(self, g_prime, new_delta, *args, **kwargs):
        """ Returns delta^{k-1}

            Notes
            -----
            Since the Split layer maps the input into 2 outputs, all we need to do to find
            delta^{k-1} is perform the reverse map of the 2 outputs
            Also note that g_prime1 and g_prime2 should be the same np.array, as this defines
            g'_{k-1}, that is the input for this layer

            Parameters
            ----------
            g_prime : tuple of (N, k) np.array
                Tuple of g'_{k-1} - left child being the first element and right child the second element
            new_delta : list of (N, k) np.array
                Note that the Split node is assumed to only have 2 children, the SplitLeftChild and SplitRightChild
                These 2 children can possible have multiple children, not limited to just 1.

            Returns
            -------
            (N, *input_shape) np.array
        """

        out_delta = np.zeros((len(new_delta[0]), *self.input_shape))

        out_delta[:, self.mask] = new_delta[0]
        out_delta[:, ~self.mask] = new_delta[1]

        return out_delta

    @check_built
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
            {}
        """

        return {}

    @check_built
    def update_parameters_(self, *args):
        """ This layer has no trainable parameters so nothing will be performed

            Notes
            -----
            Because this layer has no trainable parameters, the arguments passed
            into this method should be :obj:None, instead of np.array
        """
        pass

    @check_built
    def get_weights(self, as_dict=False):
        if as_dict:
            return {}
        return None

    @check_built
    def summary_(self):
        return 'Split Layer', f'Output Shape {(None, *self.output_shape)}'

    @property
    def left(self):
        return self.children[0]

    @property
    def right(self):
        return self.children[1]

    @property
    def activation_function_(self):
        return self.parents[0].activation_function_
