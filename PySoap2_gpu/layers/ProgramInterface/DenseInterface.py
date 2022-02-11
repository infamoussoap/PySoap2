import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array

from PySoap2_gpu.layers.c_code.dense_c_code import dense_source_code
from PySoap2_gpu.Exceptions import check_for_valid_context


class DenseInterface:
    """ The interface between the compiled pyopencl-c code with python

        Notes
        -----
        Arguments to all methods are assumed to be stored on the device
    """

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
        if DenseInterface.initialized:
            return

        DenseInterface.context = context
        DenseInterface.queue = queue

        DenseInterface.program = cl.Program(context, dense_source_code).build()

        DenseInterface.initialized = True

    @staticmethod
    def predict(z, W, b):
        check_for_valid_context(DenseInterface.context, z, W, b)

        input_length = np.int32(z.shape[1])
        output_length = np.int32(b.shape[0])
        N = len(z)

        out = cl_array.zeros(DenseInterface.queue, (N, output_length), np.float64)

        device_global_shape = out.shape
        event = DenseInterface.program.predict(DenseInterface.queue, device_global_shape,
                                               None,
                                               z.data, W.data, b.data,
                                               input_length, output_length, out.data)
        event.wait()
        return out

    @staticmethod
    def delta_back_prop(g_prime, delta, W):
        check_for_valid_context(DenseInterface.context, g_prime, delta, W)

        input_length = np.int32(g_prime.shape[1])
        output_length = np.int32(delta.shape[1])
        out = cl_array.zeros(DenseInterface.queue, g_prime.shape, np.float64)

        device_global_shape = g_prime.shape
        event = DenseInterface.program.delta_back_prop(DenseInterface.queue,
                                                       device_global_shape, None,
                                                       g_prime.data, delta.data, W.data,
                                                       input_length, output_length, out.data)
        event.wait()
        return out

    @staticmethod
    def weight_gradient(delta, prev_z):
        check_for_valid_context(DenseInterface.context, delta, prev_z)

        output_length = np.int32(delta.shape[1])
        input_length = np.int32(prev_z.shape[1])
        N = np.int32(len(prev_z))

        W_grad = cl_array.zeros(DenseInterface.queue, (output_length, input_length), np.float64)

        device_global_shape = (output_length, input_length)  # Same shape as the weight matrix
        event = DenseInterface.program.weight_gradient(DenseInterface.queue,
                                                       device_global_shape, None,
                                                       delta.data, prev_z.data,
                                                       input_length, output_length, N,
                                                       W_grad.data)
        event.wait()

        return W_grad

    @staticmethod
    def bias_gradient(delta):
        check_for_valid_context(DenseInterface.context, delta)

        N = np.int32(delta.shape[0])
        output_length = np.int32(delta.shape[1])
        b_grad = cl_array.zeros(DenseInterface.queue, (output_length,), np.float64)

        device_global_shape = (output_length,)
        event = DenseInterface.program.bias_gradient(DenseInterface.queue, device_global_shape, None,
                                                     delta.data,
                                                     output_length, N,
                                                     b_grad.data)
        event.wait()
        return b_grad
