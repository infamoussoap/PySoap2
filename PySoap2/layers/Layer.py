import abc


class Layer(abc.ABC):
    """ Abstract class to be used when creating layers for the `sequential` class
    """

    @abc.abstractmethod
    def build(self):
        """ When the build method is called in the `sequential` class it invokes this
            method. This allows the for the given layer to initialise the required variables.
        """
        pass

    @abc.abstractmethod
    def predict(self, z, output_only=True, pre_activation_of_input=None, training=False):
        """ When the predict method is called in the `sequential` class it invokes this
            method. This method is to perform the forward propagation of this current layer

            Raises
            ------
            NotBuiltError
                If the instance has not been built yet, i.e. `build` method must be called

            Notes
            -----
            Most classes that will inherent `sequential_layer` will have an associated activation
            function.
            If `output_only = True` then this method is to return only the post-activated
            output.
            If `output_only = False` then this method is will return the pre-activated and post-activated
            output, in that order.

            pre_activation_of_input should only be used when the layer is a pass-through layer, with no
            activation function (split and concatenate layers).
        """
        pass

    @abc.abstractmethod
    def get_delta_backprop_(self, g_prime, new_delta, prev_z):
        """ Returns the associated back propagation 'delta' for this layer

            Parameters
            ----------
            g_prime : np.array
                Should be the derivative of the output of the previous layer, g'_{k-1}(a^{k-1}_{m,j})
            new_delta : list of np.array
                The gradients (or deltas) from the children, to be back-propagated. Note
            prev_z : np.array
                The input for this layer, z^{n-1}

            Raises
            ------
            NotBuiltError
                If the instance has not been built yet, i.e. `build` method must be called

            Notes
            -----
            Generally speaking, the layer does not need to be built in order for this method
            to work correctly. So perhaps this should be a static method of this class, but I'm not
            too sure about that yet. But until then, NotBuiltError will be raised unless it has
            been built

            While a layer can have multiple children, they still can't have multiple parents (unless you are
            the concatenate node). As such, g_prime and prev_z are np.array, not a list of np.array.
        """
        pass

    @abc.abstractmethod
    def get_parameter_gradients_(self, delta, prev_z):
        """ Returns the gradient for the parameters of the layer, in the form of a dictionary

            Raises
            ------
            NotBuiltError
                If the instance has not been built yet, i.e. `build` method must be called
        """
        pass

    @abc.abstractmethod
    def update_parameters_(self, parameter_updates):
        """ Once all the gradients have been calculated, this method will be called
            so the current layer can update it's weights and biases

            Raises
            ------
            NotBuiltError
                If the instance has not been built yet, i.e. `build` method must be called
        """
        pass

    @abc.abstractmethod
    def get_weights(self, as_dict=False):
        """ Returns the weights/filter and bias of this layer

            Parameters
            ----------
            as_dict : bool
                If as_dict, then the weights will be returned as a dictionary

            Raises
            ------
            NotBuiltError
                If the instance has not been built yet, i.e. `build` method must be called
        """
        pass

    @abc.abstractmethod
    def summary_(self):
        """ Returns a tuple of strings that should identify the class.
            The 0th argument - The type of layer and the filter/weight shape
            The 1st argument - The output of the layer

            Raises
            ------
            NotBuiltError
                If the instance has not been built yet, i.e. `build` method must be called
        """

    @property
    @abc.abstractmethod
    def activation_function_(self):
        """ Returns the activation function of this layer
        """
        pass
