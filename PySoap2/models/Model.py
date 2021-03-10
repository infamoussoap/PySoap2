import functools

from PySoap2.optimizers import Optimizer, get_optimizer
from PySoap2 import get_error_function, get_metric_function


class Model:
    @staticmethod
    def _is_valid_model(start_layer, end_layer):
        """ Checks to see if there is a valid that connects the input layer to the output layer """
        if len(start_layer.children) == 0:
            return start_layer == end_layer
        if end_layer in start_layer.children:
            return True

        return any([Model._is_valid_model(child, end_layer) for child in start_layer.children])

    def __init__(self, input_layer, output_layer):
        if not self._is_valid_model(input_layer, output_layer):
            raise ValueError('There is no path from the input layer to the output layer.')

        self.input_layer = input_layer
        self.output_layer = output_layer

        self.optimizer = None

        self.loss_function = None
        self.metric_function = None

    def build(self, loss_function, optimizer, metrics=None):
        """ Build the layers in the tree network, and save attributes of Model """
        if isinstance(optimizer, Optimizer):
            self.optimizer = optimizer
        elif isinstance(optimizer, str):
            self.optimizer = get_optimizer(optimizer)
        else:
            raise ValueError("optimizer must be an instance of Optimizer or str")

        self.loss_function = loss_function
        self.metric_function = metrics

        for layer in self.layers_by_number_of_parents:
            layer.build()

    @property
    @functools.lru_cache()
    def layers_by_number_of_parents(self):
        """ Returns a list by order of the nodes with the least amount of parents to the most parents """
        current_layers = [self.output_layer]  # Terminal node will have the most parents
        layer_order = []

        while len(current_layers) > 0:
            for layer in current_layers:
                if layer in layer_order:
                    layer_order.remove(layer)
                layer_order.append(layer)

            parents_of_current_layers = [layer.parents for layer in current_layers]
            current_layers = set(functools.reduce(lambda x, y: x + y, parents_of_current_layers))

        return layer_order[::-1]

    def predict(self, z, output_only=True):
        """ Perform forward propagation of the whole network

            Parameters
            ----------
            z : (N, *input_shape) np.array
                The Input
            output_only : bool
                If true then only the model output will be returned
                Otherwise the pre and post activations will be returned as a dictionary

            Returns
            -------
            (N, *output_shape) np.array
                This is returned in output_only=True

            or

            dict of str - (np.array, np.array)
                This is returned in output_only=False
                The str keys are the layers memory location, i.e. their unique identifier. The associated value
                will be a tuple of the pre and post activations.
        """

        # Input is a special case
        cached_outputs = {self.input_layer.memory_location: self.input_layer.predict(z, output_only=output_only)}

        for layer in self.layers_by_number_of_parents[1:]:  # Input is assumed to have the least number of parents
            layer_id = layer.memory_location

            layer_arg = self._get_layer_predict_arguments(layer, cached_outputs, output_only=output_only)
            cached_outputs[layer_id] = layer.predict(layer_arg, output_only=output_only)

        if not output_only:
            return cached_outputs
        return cached_outputs[self.output_layer.memory_location]

    @staticmethod
    def _get_layer_predict_arguments(layer, cached_outputs, output_only=True):
        """ Returns the arguments to be passed into layer to be predicted

            Parameters
            ----------
            layer : :obj:Layer
                The layer to return the output of
            cached_outputs : dict of str - :obj:
                Stores the outputs of the parent nodes. Note that when calling this layer
                it is assumed that the the root node (or terminal nodes) is inside cached_outputs
            output_only : bool
                If true then cached_outputs is dict of str - np.array
                If false then cached_outputs is dict of str - (np.array, np.array)
        """
        layer_args = [cached_outputs[parent.memory_location] for parent in layer.parents]

        if not output_only:
            layer_args = [post_activation for (pre_activation, post_activation) in layer_args]

        if len(layer.parents) == 1:
            return layer_args[0]
        return layer_args

    def _get_layer_deltas(self, x_train, y_train):
        """ Returns the delta^k for all the layers """

        prediction = self.predict(x_train, output_only=False)

        cached_pre_activation = {key: val[0] for (key, val) in prediction.items()}
        cached_output = {key: val[1] for (key, val) in prediction.items()}
        cached_delta = {}

        output_id = self.output_layer.memory_location

        cached_delta[output_id] = self._loss_function(cached_output[output_id], y_train,
                                                      grad=True)  # Gradient of output
        if self.loss_function == 'cross_entropy':
            unique_identifier = self.output_layer.memory_location
            cached_delta[unique_identifier] = cached_output[unique_identifier] - y_train

        for layer in self.layers_by_number_of_children:
            for parent in layer.parents:
                delta = tuple([cached_delta[child.memory_location] for child in parent.children])
                if len(parent.children) == 1:
                    delta = delta[0]

                g_prime = parent.activation_function_(cached_pre_activation[parent.memory_location], grad=True)
                z = cached_output[parent.memory_location]

                cached_delta[parent.memory_location] = layer.get_delta_backprop_(g_prime, delta, z)

        return cached_delta


    @property
    @functools.lru_cache()
    def layers_by_number_of_children(self):
        """ Returns a list by order of the nodes with the least amount of children to the most children """
        current_layers = [self.input_layer]  # Input node will have the most children
        layer_order = []

        while len(current_layers) > 0:
            for layer in current_layers:
                if layer in layer_order:
                    layer_order.remove(layer)
                layer_order.append(layer)

            children_of_current_layers = [layer.children for layer in current_layers]
            current_layers = set(functools.reduce(lambda x, y: x + y, children_of_current_layers))

        return layer_order[::-1]

    @property
    def _loss_function(self):
        return get_error_function(self.loss_function)

    @property
    def _metric(self):
        if self.metric_function is not None:
            return get_metric_function(self.metric_function)
        return None
