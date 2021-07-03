import numpy as np

import pyopencl as cl
import pyopencl.array as cl_array
from pyopencl.elementwise import ElementwiseKernel

from .cl_math_functions_c_code import softmax_source_code
from .cl_array_tricks import ClArrayTricks


class ClMathFunctions:
    initialized = False

    device_context = None
    device_queue = None

    relu_program = None
    relu_grad_program = None
    sigmoid_program = None
    softmax_gpu_program = None

    def __init__(self, device_context, device_queue):
        # If this class is initialized, it means that the programs is already on the device
        if ClMathFunctions.initialized:
            return

        ClMathFunctions.device_context = device_context
        ClMathFunctions.device_queue = device_queue

        ClMathFunctions.relu_program = ElementwiseKernel(device_context, "float *x, float *out",
                                                         "out[i] = x[i] > 0 ? x[i] : 0.0", "relu")

        ClMathFunctions.relu_grad_program = ElementwiseKernel(device_context, "float *x, float *out",
                                                              "out[i] = x[i] > 0 ? 1.0 : 0.0", "relu")

        ClMathFunctions.sigmoid_program = ElementwiseKernel(device_context,
                                                            "float *x, float *out",
                                                            "out[i] = SIGMOID(x[i])",
                                                            "sigmoid",
                                                            preamble='#define SIGMOID(x) x > 0 ? '
                                                                     '1.0/(1.0 + exp(-x)) : exp(x) / (exp(x) + 1.0)'
                                                            )

        ClMathFunctions.softmax_gpu_program = cl.Program(device_context, softmax_source_code).build()

        ClMathFunctions.initialized = True

        if not ClArrayTricks.initialized:
            ClArrayTricks(self.device_context, self.device_queue)

    @staticmethod
    def relu(x_gpu):
        out_gpu = cl_array.empty_like(x_gpu)
        ClMathFunctions.relu_program(x_gpu, out_gpu)
        return out_gpu

    @staticmethod
    def relu_grad(x_gpu):
        out_gpu = cl_array.empty_like(x_gpu)
        ClMathFunctions.relu_grad_program(x_gpu, out_gpu)
        return out_gpu

    @staticmethod
    def sigmoid(x_gpu):
        out_gpu = cl_array.empty_like(x_gpu)
        ClMathFunctions.sigmoid_program(x_gpu, out_gpu)
        return out_gpu

    @staticmethod
    def softmax(x_gpu):
        input_length_gpu = cl_array.to_device(ClMathFunctions.device_queue, np.array(x_gpu.shape[-1], dtype=np.int32))
        max_val_gpu = ClArrayTricks.max_across_last_axis(x_gpu)

        softmax_val = cl_array.empty_like(x_gpu)

        global_shape_col = x_gpu.shape[-1]
        global_shape_row = np.prod(x_gpu.shape[:-1]).astype(np.int32)
        global_shape = (global_shape_row, global_shape_col)

        event = ClMathFunctions.softmax_gpu_program.softmax(ClMathFunctions.device_queue, global_shape, None,
                                                            x_gpu.data, max_val_gpu.data, input_length_gpu.data,
                                                            softmax_val.data)
        event.wait()

        return softmax_val
