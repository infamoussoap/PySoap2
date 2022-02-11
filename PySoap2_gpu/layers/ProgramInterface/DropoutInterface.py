import pyopencl as cl

from PySoap2_gpu.layers.c_code.dropout_c_code import dropout_source_code
from PySoap2_gpu.Exceptions import check_for_valid_context


class DropoutInterface:
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
        if DropoutInterface.initialized:
            return

        DropoutInterface.context = context
        DropoutInterface.queue = queue

        DropoutInterface.program = cl.Program(context, dropout_source_code).build()

        DropoutInterface.initialized = True

    @staticmethod
    def dropout(z, mask, output_length, out):
        check_for_valid_context(DropoutInterface.context, z, mask, out)

        device_global_shape = (len(z), output_length)
        event = DropoutInterface.program.dropout(DropoutInterface.queue,
                                                 device_global_shape, None,
                                                 z.data, mask.data, output_length, out.data)
        event.wait()
