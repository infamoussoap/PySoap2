from PySoap2.optimizers import Optimizer, get_optimizer


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
        if isinstance(optimizer, Optimizer):
            self.optimizer = optimizer
        elif isinstance(optimizer, str):
            self.optimizer = get_optimizer(optimizer)
        else:
            raise ValueError("optimizer must be an instance of Optimizer or str")

        self.loss_function = loss_function
        self.metric_function = metrics

        self._build_layer(self.output_layer)

    @staticmethod
    def _build_layer(layer):
        for parent in layer.parents:
            if not parent.built:
                Model._build_layer(parent)
        layer.build()

    def predict(self, z, output_only=True):
        cached_outputs = {self.input_layer.memory_location: self.input_layer.predict(z, output_only=output_only)}

        model_prediction = self._propagated_output_of_layer(self.output_layer, cached_outputs, output_only=output_only)
        if not output_only:
            return cached_outputs
        return model_prediction

    @staticmethod
    def _propagated_output_of_layer(layer, cached_outputs, output_only=True):
        unique_identifier = layer.memory_location

        if unique_identifier not in cached_outputs:
            if len(layer.parents) == 0: 
                raise ValueError(f'{layer} has no parent nodes')

            layer_arg = Model._get_layer_argument(layer, cached_outputs, output_only=output_only)
            cached_outputs[unique_identifier] = layer.predict(layer_arg, output_only=output_only)

        return cached_outputs[unique_identifier]

    @staticmethod
    def _get_layer_argument(layer, cached_outputs, output_only=True):
        outputs_of_parents = [Model._propagated_output_of_layer(parent, cached_outputs, output_only=output_only)
                              for parent in layer.parents]

        if not output_only:
            layer_args = [post_activation for (pre_activation, post_activation) in outputs_of_parents]
        else:
            layer_args = outputs_of_parents

        if len(outputs_of_parents) == 1:
            return layer_args[0]
        return layer_args
