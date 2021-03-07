from PySoap2 import get_activation_function


class LayerBaseAttributes:
    """ Define base attributes of Layer

        Attributes
        ----------
        input_shape : tuple
        output_shape : tuple
        activation_function : str
            Assumed to be linear activation
        built : bool
            Set to False
    """
    def __init__(self):
        self.input_shape = ()
        self.output_shape = ()

        self.activation_function = 'linear'

        self.built = False

    @property
    def activation_function_(self):
        return get_activation_function(self.activation_function)

    @property
    def memory_location(self):
        """ Return the location in memory, to be used as a unique identify for instances
            of LayerBaseAttributes
        """
        return hex(id(self))
