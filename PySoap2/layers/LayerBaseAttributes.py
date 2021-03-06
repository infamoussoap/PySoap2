class LayerBaseAttributes:
    """ Define base attributes of Layer

        Attributes
        ----------
        input_shape : tuple
            Empty Tuple
        output_shape : tuple
            Empty Tuple
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
