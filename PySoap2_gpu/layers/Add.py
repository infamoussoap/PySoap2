from PySoap2_gpu.layers import Layer
from PySoap2_gpu.layers.NetworkNode import NetworkNode
from PySoap2_gpu.layers.LayerBaseAttributes import LayerBaseAttributes

from PySoap2.layers import Add as AddCPU


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
