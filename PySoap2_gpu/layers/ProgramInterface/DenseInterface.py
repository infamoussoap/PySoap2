import pyopencl as cl

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
    def predict(z, W, b, input_length, output_length, out):
        check_for_valid_context(DenseInterface.context, z, W, b, out)

        device_global_shape = out.shape
        event = DenseInterface.program.predict(DenseInterface.queue, device_global_shape,
                                               None,
                                               z.data, W.data, b.data, input_length,
                                               output_length, out.data)
        event.wait()

    @staticmethod
    def delta_back_prop(g_prime, new_delta, W, input_length, output_length, out):
        check_for_valid_context(DenseInterface.context, g_prime, new_delta, W, out)

        device_global_shape = g_prime.shape
        event = DenseInterface.program.delta_back_prop(DenseInterface.queue,
                                                       device_global_shape, None,
                                                       g_prime.data, new_delta.data, W.data,
                                                       input_length, output_length, out.data)
        event.wait()

    @staticmethod
    def weight_gradient(delta, prev_z, input_length, output_length, N, out):
        check_for_valid_context(DenseInterface.context, delta, prev_z, out)

        device_global_shape = (output_length, input_length)  # Same shape as the weight matrix
        event = DenseInterface.program.weight_gradient(DenseInterface.queue,
                                                       device_global_shape, None,
                                                       delta.data, prev_z.data, input_length,
                                                       output_length, N, out.data)
        event.wait()

    @staticmethod
    def bias_gradient(delta, output_length, N, out):
        check_for_valid_context(DenseInterface.context, delta, out)

        device_global_shape = (output_length,)
        event = DenseInterface.program.bias_gradient(DenseInterface.queue,
                                                     device_global_shape, None, delta.data,
                                                     output_length, N, out.data)
        event.wait()
