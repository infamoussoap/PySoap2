import numpy as np

import pyopencl as cl
import pyopencl.array as cl_array

from PySoap2_gpu.layers.c_code.conv2d_c_code import conv2d_source_code


class Conv2DInterface:
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
        if Conv2DInterface.initialized:
            return

        Conv2DInterface.context = context
        Conv2DInterface.queue = queue

        Conv2DInterface.program = cl.Program(context, conv2d_source_code).build()

        Conv2DInterface.initialized = True

    @staticmethod
    def predict(z, filter_, bias_, stride):
        program = Conv2DInterface.program
        queue = Conv2DInterface.queue

        input_shape = [np.int32(x) for x in z.shape[1:]]

        n = np.int32((input_shape[0] - filter_.shape[0]) / stride + 1)
        m = np.int32((input_shape[1] - filter_.shape[1]) / stride + 1)
        output_shape = (n, m, np.int32(filter_.shape[3]))
        out = cl_array.zeros(queue, (len(z), *output_shape), np.float64)

        filter_shape = [np.int32(x) for x in filter_.shape]

        filter_height, filter_width, _, num_of_filters = filter_shape
        _, image_width, image_depth = input_shape
        output_height, output_width, _ = output_shape
        input_length = np.int32(np.prod(input_shape))
        output_length = np.int32(np.prod(output_shape))

        events = []
        for i in range(num_of_filters):
            current_filter = np.int32(i)

            event = program.predict(queue, (len(z), output_height, output_width), None,
                                    z.data, filter_.data, bias_.data,
                                    filter_height, filter_width, num_of_filters,
                                    stride, current_filter,
                                    image_width, image_depth,
                                    output_width,
                                    input_length, output_length,
                                    out.data)

            events.append(event)

        cl.wait_for_events(events)

        return out

    @staticmethod
    def delta_back_prop(delta, eye_conv, g_prime, input_length, output_length, out):
        program = Conv2DInterface.program
        queue = Conv2DInterface.queue

        global_shape = (np.prod(out.shape),)

        event = program.delta_back_prop(queue, global_shape, None,
                                        delta.data, eye_conv.data, g_prime.data, input_length, output_length, out.data)

        event.wait()
        return out

    @staticmethod
    def filter_gradient(prev_z, delta,
                        output_height, output_width, num_of_filters,
                        stride,
                        image_width, image_depth,
                        N, input_length,
                        filter_width,
                        out):
        program = Conv2DInterface.program
        queue = Conv2DInterface.queue

        global_shape = (np.prod(out.shape),)

        event = program.filter_gradient(queue, global_shape, None,
                                        prev_z.data, delta.data,
                                        output_height, output_width, num_of_filters,
                                        stride,
                                        image_width, image_depth,
                                        N, input_length,
                                        filter_width,
                                        out.data)
        event.wait()

    @staticmethod
    def bias_gradient(delta, sum_length, num_of_filters, out):
        program = Conv2DInterface.program
        queue = Conv2DInterface.queue

        global_shape = out.shape

        event = program.bias_gradient(queue, global_shape, None,
                                      delta.data, sum_length, num_of_filters, out.data)
        event.wait()
