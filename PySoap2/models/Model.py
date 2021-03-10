from PySoap2.optimizers import Optimizer, get_optimizer
import functools


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
