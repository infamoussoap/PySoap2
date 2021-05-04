import numpy as np

import pyopencl as cl
import pyopencl.array as cl_array

from PySoap2_gpu.layers import Layer
from PySoap2_gpu.layers.NetworkNode import NetworkNode
from PySoap2_gpu.layers.LayerBaseAttributes import LayerBaseAttributes

from .c_code.split_c_code import split_source_code


class SplitChild(NetworkNode, LayerBaseAttributes, Layer):
    def __init__(self, mask):
        LayerBaseAttributes.__init__(self)
        NetworkNode.__init__(self)

        self.mask = mask

    def build(self, device_context, device_queue):
        pass




