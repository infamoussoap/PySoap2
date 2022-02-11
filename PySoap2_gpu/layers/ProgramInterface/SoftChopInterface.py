import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array

from PySoap2_gpu.layers.c_code.multi_softchop_c_code import multi_softchop_source_code
from PySoap2_gpu.layers.c_code.softchop_c_code import softchop_source_code

from PySoap2_gpu.Exceptions import check_for_valid_context


class MultiSoftChopInterface:
    """ The interface between the compiled pyopencl-c code with python

        Notes
        -----
        Arguments to all methods are assumed to be stored on the device
    """

    device_context = None
    device_queue = None

    device_program = None

    initialized = False

    def __init__(self, device_context, device_queue):
        if MultiSoftChopInterface.initialized:
            return

        MultiSoftChopInterface.device_context = device_context
        MultiSoftChopInterface.device_queue = device_queue

        MultiSoftChopInterface.device_program = cl.Program(device_context, multi_softchop_source_code).build()

        MultiSoftChopInterface.initialized = True

    @staticmethod
    def eval(x_device, a1_device, a2_device, epsilon1_device, epsilon2_device):
        check_for_valid_context(MultiSoftChopInterface.device_context, x_device, a1_device, a2_device,
                                epsilon1_device, epsilon2_device)

        out_device = cl_array.empty_like(x_device)

        input_length = np.array(np.prod(a1_device.shape)).astype(np.int32)
        N = np.array(len(x_device)).astype(np.int32)

        input_length_device = cl_array.to_device(MultiSoftChopInterface.device_queue, input_length)

        args = [x_device, a1_device, a2_device, epsilon1_device, epsilon2_device, input_length_device, out_device]
        args_data = [arg.data for arg in args]

        event = MultiSoftChopInterface.device_program.softchop_eval(MultiSoftChopInterface.device_queue, (N, input_length), None,
                                                                    *args_data)
        event.wait()

        return out_device

    @staticmethod
    def dx(x_device, a1_device, a2_device, epsilon1_device, epsilon2_device):
        check_for_valid_context(MultiSoftChopInterface.device_context, x_device, a1_device, a2_device,
                                epsilon1_device, epsilon2_device)

        out_device = cl_array.empty_like(x_device)

        input_length = np.array(np.prod(a1_device.shape)).astype(np.int32)
        N = np.array(len(x_device)).astype(np.int32)

        input_length_device = cl_array.to_device(MultiSoftChopInterface.device_queue, input_length)

        args = [x_device, a1_device, a2_device, epsilon1_device, epsilon2_device, input_length_device, out_device]
        args_data = [arg.data for arg in args]

        event = MultiSoftChopInterface.device_program.softchop_dx(MultiSoftChopInterface.device_queue, (N, input_length), None,
                                                                  *args_data)
        event.wait()

        return out_device

    @staticmethod
    def da1(x_device, a1_device, a2_device, epsilon1_device, epsilon2_device):
        check_for_valid_context(MultiSoftChopInterface.device_context, x_device, a1_device, a2_device,
                                epsilon1_device, epsilon2_device)

        out_device = cl_array.empty_like(x_device)

        input_length = np.array(np.prod(a1_device.shape)).astype(np.int32)
        N = np.array(len(x_device)).astype(np.int32)

        input_length_device = cl_array.to_device(MultiSoftChopInterface.device_queue, input_length)

        args = [x_device, a1_device, a2_device, epsilon1_device, epsilon2_device, input_length_device, out_device]
        args_data = [arg.data for arg in args]

        event = MultiSoftChopInterface.device_program.softchop_da1(MultiSoftChopInterface.device_queue, (N, input_length), None,
                                                                   *args_data)
        event.wait()

        return out_device

    @staticmethod
    def da2(x_device, a1_device, a2_device, epsilon1_device, epsilon2_device):
        check_for_valid_context(MultiSoftChopInterface.device_context, x_device, a1_device, a2_device,
                                epsilon1_device, epsilon2_device)

        out_device = cl_array.empty_like(x_device)

        input_length = np.array(np.prod(a1_device.shape)).astype(np.int32)
        N = np.array(len(x_device)).astype(np.int32)

        input_length_device = cl_array.to_device(MultiSoftChopInterface.device_queue, input_length)

        args = [x_device, a1_device, a2_device, epsilon1_device, epsilon2_device, input_length_device, out_device]
        args_data = [arg.data for arg in args]

        event = MultiSoftChopInterface.device_program.softchop_da2(MultiSoftChopInterface.device_queue, (N, input_length), None,
                                                                   *args_data)
        event.wait()

        return out_device

    @staticmethod
    def de1(x_device, a1_device, a2_device, epsilon1_device, epsilon2_device):
        check_for_valid_context(MultiSoftChopInterface.device_context, x_device, a1_device, a2_device,
                                epsilon1_device, epsilon2_device)

        out_device = cl_array.empty_like(x_device)

        input_length = np.array(np.prod(a1_device.shape)).astype(np.int32)
        N = np.array(len(x_device)).astype(np.int32)

        input_length_device = cl_array.to_device(MultiSoftChopInterface.device_queue, input_length)

        args = [x_device, a1_device, a2_device, epsilon1_device, epsilon2_device, input_length_device, out_device]
        args_data = [arg.data for arg in args]

        event = MultiSoftChopInterface.device_program.softchop_de1(MultiSoftChopInterface.device_queue, (N, input_length), None,
                                                                   *args_data)
        event.wait()

        return out_device

    @staticmethod
    def de2(x_device, a1_device, a2_device, epsilon1_device, epsilon2_device):
        check_for_valid_context(MultiSoftChopInterface.device_context, x_device, a1_device, a2_device,
                                epsilon1_device, epsilon2_device)

        out_device = cl_array.empty_like(x_device)

        input_length = np.array(np.prod(a1_device.shape)).astype(np.int32)
        N = np.array(len(x_device)).astype(np.int32)

        input_length_device = cl_array.to_device(MultiSoftChopInterface.device_queue, input_length)

        args = [x_device, a1_device, a2_device, epsilon1_device, epsilon2_device, input_length_device, out_device]
        args_data = [arg.data for arg in args]

        event = MultiSoftChopInterface.device_program.softchop_de2(MultiSoftChopInterface.device_queue, (N, input_length), None,
                                                                   *args_data)
        event.wait()

        return out_device


class SoftChopInterface:
    context = None
    queue = None

    program = None

    initialized = False

    def __init__(self, device_context, device_queue):
        if SoftChopInterface.initialized:
            return

        SoftChopInterface.context = device_context
        SoftChopInterface.queue = device_queue

        SoftChopInterface.program = cl.Program(device_context, softchop_source_code).build()

        SoftChopInterface.initialized = True

    @staticmethod
    def delta_back_prop(g_prime, new_delta, dz, out):
        check_for_valid_context(SoftChopInterface.context,
                                g_prime, new_delta, dz, out)

        device_global_shape = (int(np.prod(g_prime.shape)),)
        event = SoftChopInterface.program.delta_back_prop(SoftChopInterface.queue,
                                                          device_global_shape, None,
                                                          g_prime.data, new_delta.data, dz.data,
                                                          out.data)
        event.wait()

    @staticmethod
    def parameter_gradient(delta, parameter, input_length, N, out):
        check_for_valid_context(SoftChopInterface.context,
                                delta, parameter, out)

        device_global_shape = (input_length,)
        event = SoftChopInterface.program.parameter_gradient(SoftChopInterface.queue,
                                                             device_global_shape, None,
                                                             delta.data, parameter.data,
                                                             input_length, N, out.data)
        event.wait()
