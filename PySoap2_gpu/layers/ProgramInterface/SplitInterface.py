import numpy as np

import pyopencl as cl
import pyopencl.array as cl_array

from PySoap2_gpu.layers.c_code.split_c_code import split_source_code
from PySoap2_gpu.Exceptions import check_for_valid_context


class SplitInterface:
    context = None
    queue = None

    program = None

    initialized = False

    def __init__(self, device_context, device_queue):
        if SplitInterface.initialized:
            return

        SplitInterface.context = device_context
        SplitInterface.queue = device_queue

        SplitInterface.program = cl.Program(device_context, split_source_code).build()

        SplitInterface.initialized = True

    @staticmethod
    def get_input_at_mask(input_, mask_positions, output_shape):
        check_for_valid_context(SplitInterface.context, input_, mask_positions)

        N = input_.shape[0]
        out = cl_array.zeros(SplitInterface.queue, (N, *output_shape), np.float64)

        output_length = np.int32(output_shape)
        input_length = np.int32(np.prod(input_.shape[1:]))

        device_global_shape = (N, output_length)
        event = SplitInterface.program.get_input_at_mask(SplitInterface.queue,
                                                         device_global_shape, None,
                                                         input_.data, mask_positions.data,
                                                         input_length,
                                                         output_length, out.data)
        event.wait()
        return out

    @staticmethod
    def set_input_at_mask_as_output(input_, mask_positions, input_length, output_length, output_):
        check_for_valid_context(SplitInterface.context, input_, mask_positions, output_)

        N, *input_shape = output_.shape
        device_global_shape = (N, int(np.prod(input_shape)))

        event = SplitInterface.program.set_input_at_mask_as_output(SplitInterface.queue,
                                                                   device_global_shape, None,
                                                                   input_.data, mask_positions.data,
                                                                   input_length,
                                                                   output_length, output_.data)
        event.wait()
