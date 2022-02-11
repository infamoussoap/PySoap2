import numpy as np

import pyopencl as cl
import pyopencl.array as cl_array

from PySoap2_gpu.layers.c_code.maxpooling_c_code import maxpool_source_code


class MaxPoolingInterface:
    context = None
    queue = None

    program = None

    initialized = False

    def __init__(self, context, queue):
        """ Compile the c-program

            Notes
            -----
            Once this class has been initialized, the c-program will be compiled on the given device context and
            will be bound to the class (not instances of the class).
            It will no longer be possible to re-initialize this class again.
        """
        if MaxPoolingInterface.initialized:
            return

        MaxPoolingInterface.context = context
        MaxPoolingInterface.queue = queue

        MaxPoolingInterface.program = cl.Program(context, maxpool_source_code).build()

        MaxPoolingInterface.initialized = True

    @staticmethod
    def maxpool_2d(z, window_shape, stride):
        queue = MaxPoolingInterface.queue
        program = MaxPoolingInterface.program

        image_shape = z.shape[1:]
        n = np.int32((image_shape[0] - window_shape[0]) / stride + 1)
        m = np.int32((image_shape[1] - window_shape[1]) / stride + 1)
        output_shape = (n, m, image_shape[2])

        max_out = cl_array.zeros(queue, (len(z), *output_shape), np.float64)
        argmax_out = cl_array.zeros(queue, (len(z), *output_shape), np.int32)

        image_shape = [np.int32(x) for x in image_shape]
        output_shape = [np.int32(x) for x in output_shape]
        window_shape = [np.int32(x) for x in window_shape]

        image_length = np.int32(np.prod(image_shape))
        output_length = np.int32(np.prod(output_shape))

        events = []
        global_shape = (len(z), n, m)
        for current_channel in range(image_shape[2]):
            event = program.maxpool_2d(queue, global_shape, None,
                                       z.data,
                                       *image_shape,
                                       *window_shape, np.int32(stride),
                                       np.int32(current_channel),
                                       n, m,
                                       image_length, output_length,
                                       max_out.data, argmax_out.data)
            events.append(event)

        cl.wait_for_events(events)
        return max_out, argmax_out

    @staticmethod
    def add_at(z, index, vals):
        queue = MaxPoolingInterface.queue
        program = MaxPoolingInterface.program

        global_shape = (int(np.prod(vals.shape)),)
        event = program.add_at(queue, global_shape, None,
                               z.data, index.data, vals.data)
        event.wait()
        return z
