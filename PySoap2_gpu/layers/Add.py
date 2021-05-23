from PySoap2_gpu.layers import Layer
from PySoap2_gpu.layers.NetworkNode import NetworkNode
from PySoap2_gpu.layers.LayerBaseAttributes import LayerBaseAttributes

from PySoap2.layers import Add as AddCPU

from PySoap2_gpu import ActivationFunction


class Add(AddCPU, NetworkNode, LayerBaseAttributes, Layer):
    """ Adds the inputs to this layer
    """

    def __init__(self):
        """ A fully connected layer """
        NetworkNode.__init__(self)
        LayerBaseAttributes.__init__(self)
        AddCPU.__init__(self)

    def build(self, device_context, device_queue):
        """ Initialises the weight and bias units """
        self.device_context = device_context
        self.device_queue = device_queue

        super().build()

    @property
    def activation_function_(self):
        """ Need to overload parents activation function to use the gpu-programs """
        if not ActivationFunction.initialized:
            ActivationFunction(self.device_context, self.device_queue)
        return ActivationFunction.get_activation_function(self.activation_function)
