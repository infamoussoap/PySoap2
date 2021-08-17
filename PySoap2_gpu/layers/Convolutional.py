import numpy as np
from functools import reduce

import pyopencl as cl
import pyopencl.array as cl_array

from PySoap2_gpu.layers import Layer
from PySoap2_gpu.layers import NetworkNode
from PySoap2_gpu.layers.LayerBaseAttributes import LayerBaseAttributes

from .c_code.conv2d_c_code import conv2d_source_code
from .ValueChecks import assert_instance_of_cl_array
from .ValueChecks import check_built

from PySoap2_gpu.Exceptions import check_for_valid_context


class Conv2DInterfaceToDevice:
    device_context = None
    device_queue = None

    device_program = None

    initialized = False

    def __init__(self, device_context, device_queue):
        """ Compile the c-program

            Notes
            -----
            Once this class has been initialized, the c-program will be compiled on the given device context and
            will be bound to the class (not instances of the class).
            It will no longer be possible to re-initialize this class again.
        """
        if Conv2DInterfaceToDevice.initialized:
            return

        Conv2DInterfaceToDevice.device_context = device_context
        Conv2DInterfaceToDevice.device_queue = device_queue

        Conv2DInterfaceToDevice.device_program = cl.Program(device_context, conv2d_source_code).build()

        Conv2DInterfaceToDevice.initialized = True

    @staticmethod
    def predict(z, filter_, bias_, filter_height, filter_width, num_of_filters, stride, ):
        program = Conv2DInterfaceToDevice.device_program
        queue = Conv2DInterfaceToDevice.device_queue

        out_new = cl_array.zeros(queue, (len(z), height, width, num_filter), np.float64)
        for i in range(num_of_filters):
            current_filter = np.int32(i)

            event = program.conv2d(queue, (len(z), height, width), None,
                                   z.data, filter_.data, bias_.data,
                                   filter_height, filter_width, num_of_filters,
                                   stride, current_filter,
                                   image_width, image_depth,
                                   output_width,
                                   input_length, output_length,
                                   out_new.data)
        queue.finish()
        np.testing.assert_almost_equal(out_cpu, out_new.get())

    @staticmethod
    def delta_back_prop():
        pass

    @staticmethod
    def filter_gradient():
        pass

    @staticmethod
    def bias_gradient():
        pass

