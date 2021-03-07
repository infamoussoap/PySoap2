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

        self.build_layer_(self.output_layer)

    @staticmethod
    def build_layer_(layer):
        for parent in layer.parents:
            if not parent.built:
                Model.build_layer_(parent)
        layer.build()
