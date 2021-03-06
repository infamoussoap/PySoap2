class Model:

    @staticmethod
    def _is_valid_model(input_layer, output_layer):
        """ Checks to see if there is a valid that connects the input layer to the output layer """
        if len(input_layer.children) == 0:
            return input_layer == output_layer

        return any([Model._is_valid_model(child, output_layer) for child in input_layer.children])

    def __init__(self, input_layer, output_layer):
        self.input_layer = input_layer
        self.output_layer = output_layer
