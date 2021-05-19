from PySoap2_gpu import ActivationFunction


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
        device_context : pyopencl.Context
        device_queue : pyopencl.CommandQueue
    """
    def __init__(self):
        self.input_shape = ()
        self.output_shape = ()

        self.activation_function = 'linear'

        self.device_context = None
        self.device_queue = None

        self.built = False

    @property
    def activation_function_(self):
        if not ActivationFunction.initialized:
            ActivationFunction(self.device_context, self.device_queue)
        return ActivationFunction.get_activation_function(self.activation_function)

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
