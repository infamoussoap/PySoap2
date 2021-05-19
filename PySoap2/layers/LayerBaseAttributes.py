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
    def _memory_location(self):
        """ Return the location in memory, to be used as a unique identify for instances
            of LayerBaseAttributes

            Notes
            -----
            I'm not sure if there is anything wrong with using memory as a unique identifier,
            but another alternative, if needed, is to use an attribute bound to this class
        """
        return hex(id(self))

    @property
    def id(self):
        """ We'll use the memory location as the id of a layer """
        return f'{type(self).__name__}_{self._memory_location}'

    def get_layer_attributes_(self):
        """ Returns the attributes of the layer as a dictionary.

            Note the attributes that are returned are only the ones that can
            be saved as a hdf5 file (or converted to be saved as as hdf5)
        """
        layer_attributes = self.__dict__.copy()

        del layer_attributes['parents']
        del layer_attributes['children']

        return layer_attributes
