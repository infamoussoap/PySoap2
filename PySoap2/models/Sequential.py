from PySoap2.models import Model


class Sequential(Model):
    def __init__(self):
        super().__init__(None, None)

        self.input_layer = None
        self.current_layer = None
        self.output_layer = None

    def add(self, layer):
        if self.input_layer is None:
            self.input_layer = layer
            self.current_layer = layer
        else:
            self.current_layer = layer(self.current_layer)

    def build(self, *args, **kwargs):
        self.output_layer = self.current_layer
        super().build(*args, **kwargs)

