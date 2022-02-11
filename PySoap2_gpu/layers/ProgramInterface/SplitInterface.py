import numpy as np

import pyopencl as cl

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
    def get_input_at_mask(input_, mask_positions, input_length, output_length, output_):
        check_for_valid_context(SplitInterface.context, input_, mask_positions, output_)

        device_global_shape = output_.shape
        event = SplitInterface.program.get_input_at_mask(SplitInterface.queue,
                                                         device_global_shape, None,
                                                         input_.data, mask_positions.data,
                                                         input_length,
                                                         output_length, output_.data)
        event.wait()

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