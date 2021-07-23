import numpy as np

import pyopencl as cl
import pyopencl.array as cl_array
from pyopencl.elementwise import ElementwiseKernel

from .cl_math_functions_c_code import softmax_source_code
from .cl_math_functions_c_code import log_softmax_source_code
from .cl_array_tricks import ClArrayTricks

from PySoap2_gpu.Exceptions import check_for_valid_context


class ClMathFunctions:
    initialized = False

    device_context = None
    device_queue = None

    relu_program = None
    relu_grad_program = None
    sigmoid_program = None
    softmax_program = None
    log_softmax_program = None

    def __init__(self, device_context, device_queue):
        # If this class is initialized, it means that the programs is already on the device

        if not ClArrayTricks.initialized:
            ClArrayTricks(device_context, device_queue)

        if ClMathFunctions.initialized:
            return

        ClMathFunctions.device_context = device_context
        ClMathFunctions.device_queue = device_queue

        ClMathFunctions.relu_program = ElementwiseKernel(device_context, "double *x, double *out",
                                                         "out[i] = x[i] > 0 ? x[i] : 0.0", "relu")

        ClMathFunctions.relu_grad_program = ElementwiseKernel(device_context, "double *x, double *out",
                                                              "out[i] = x[i] > 0 ? 1.0 : 0.0", "relu")

        ClMathFunctions.sigmoid_program = ElementwiseKernel(device_context,
                                                            "double *x, double *out",
                                                            "out[i] = SIGMOID(x[i])",
                                                            "sigmoid",
                                                            preamble='#define SIGMOID(x) x > 0 ? '
                                                                     '1.0/(1.0 + exp(-x)) : exp(x) / (exp(x) + 1.0)'
                                                            )

        ClMathFunctions.softmax_program = cl.Program(device_context, softmax_source_code).build()
        ClMathFunctions.log_softmax_program = cl.Program(device_context, log_softmax_source_code).build()

        ClMathFunctions.initialized = True

    @staticmethod
    def relu(x_gpu):
        check_for_valid_context(ClMathFunctions.device_context, x_gpu)

        out_gpu = cl_array.empty_like(x_gpu)
        ClMathFunctions.relu_program(x_gpu, out_gpu)
        return out_gpu

    @staticmethod
    def relu_grad(x_gpu):
        check_for_valid_context(ClMathFunctions.device_context, x_gpu)

        out_gpu = cl_array.empty_like(x_gpu)
        ClMathFunctions.relu_grad_program(x_gpu, out_gpu)
        return out_gpu

    @staticmethod
    def sigmoid(x_gpu):
        check_for_valid_context(ClMathFunctions.device_context, x_gpu)

        out_gpu = cl_array.empty_like(x_gpu)
        ClMathFunctions.sigmoid_program(x_gpu, out_gpu)
        return out_gpu

    @staticmethod
    def softmax(x_gpu):
        check_for_valid_context(ClMathFunctions.device_context, x_gpu)

        input_length = np.int32(x_gpu.shape[-1])
        max_val_gpu = ClArrayTricks.max_across_last_axis(x_gpu)

        softmax_val = cl_array.empty_like(x_gpu)

        global_shape_col = x_gpu.shape[-1]
        global_shape_row = np.prod(x_gpu.shape[:-1]).astype(np.int32)
        global_shape = (global_shape_row, global_shape_col)

        event = ClMathFunctions.softmax_program.softmax(ClMathFunctions.device_queue, global_shape, None,
                                                        x_gpu.data, max_val_gpu.data, input_length,
                                                        softmax_val.data)
        event.wait()

        return softmax_val

    @staticmethod
    def log_softmax(x_device):
        check_for_valid_context(ClMathFunctions.device_context, x_device)

        """ x_device assumed to be (n, m) cl_array """
        max_across_last_axis = ClArrayTricks.max_across_last_axis(x_device)
        input_length = np.int32(x_device.shape[-1])
        out = cl_array.zeros_like(x_device)

        event = ClMathFunctions.log_softmax_program.log_softmax(ClMathFunctions.device_queue, x_device.shape, None,
                                                                x_device.data, max_across_last_axis.data,
                                                                input_length, out.data)
        event.wait()

        return out