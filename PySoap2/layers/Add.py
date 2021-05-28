from functools import reduce

from PySoap2.layers import Layer
from PySoap2.layers.NetworkNode import NetworkNode
from PySoap2.layers.LayerBaseAttributes import LayerBaseAttributes

from .LayerBuiltChecks import check_built


class Add(NetworkNode, LayerBaseAttributes, Layer):
    """ Add the input values

        Notes
        -----
        The activation function is simply the linear function
    """
    def __init__(self):
        NetworkNode.__init__(self)
        LayerBaseAttributes.__init__(self)

    def build(self):
        """ Initialises the weight and bias units """
        first_output_shape = self.parents[0].output_shape
        if any([first_output_shape != parent.output_shape for parent in self.parents]):
            raise ValueError('Inputs to Add Layer must have the same shape.')

        self.input_shape = tuple(parent.output_shape for parent in self.parents)
        self.output_shape = self.input_shape[0]

        self.built = True

    @check_built
    def predict(self, z, output_only=True, pre_activation_of_input=None):
        out = reduce(lambda x, y: x + y, z)

        if output_only:
            return out
        return out, out

    @check_built
    def get_delta_backprop_(self, g_prime, new_delta, prev_z):
        delta = reduce(lambda x, y: x + y, new_delta)
        return delta

    @check_built
    def get_parameter_gradients_(self, delta, prev_z):
        return {}

    @check_built
    def update_parameters_(self, parameter_updates):
        pass

    @check_built
    def get_weights(self, as_dict=False):
        if as_dict:
            return {}
        return None

    @check_built
    def summary_(self):
        return 'Add', f'Output Shape {self.input_shape}'

    def __call__(self, input_layers):
        """ __call__ of NetworkNode is overloaded as it is now assumed the parameter
            is a list of NetworkNode

            Parameters
            ----------
            input_layers : list of NetworkNode
        """
        for parent_node in input_layers:
            parent_node.add_child(self)

        self.add_parents(tuple(input_layers))

        return self
