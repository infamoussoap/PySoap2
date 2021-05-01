import numpy as np

import pyopencl as cl
import pyopencl.array as cl_array

from .cl_math_functions_c_code import softmax_source_code
from .cl_math_functions_c_code import cl_array_max_source_code


def elementwise_relu(device_context):
    return cl.elementwise.ElementwiseKernel(device_context, "float *x, float *out",
                                            "out[i] = x[i] > 0 ? x[i] : 0.0", "relu")


def elementwise_relu_grad(device_context):
    return cl.elementwise.ElementwiseKernel(device_context, "float *x, float *out",
                                            "out[i] = x[i] > 0 ? 1.0 : 0.0", "relu")


def elementwise_sigmoid(device_context):
    return cl.elementwise.ElementwiseKernel(device_context,
                                            "float *x, float *out",
                                            "out[i] = SIGMOID(x[i])",
                                            "sigmoid",
                                            preamble='#define SIGMOID(x) x > 0 ? '
                                                     '1.0/(1.0 + exp(-x)) : exp(x) / (exp(x) + 1.0)'
                                            )


def softmax_gpu(device_context, device_queue, x_gpu, grad=False):
    softmax_gpu_program = cl.Program(device_context, softmax_source_code).build()

    if grad:
        softmax_val = softmax_gpu(device_context, device_queue, x_gpu, grad=False)
        return softmax_val * (1 - softmax_val)

    input_length_gpu = cl_array.to_device(device_queue, np.array(x_gpu.shape[-1], dtype=np.int32))
    max_val_gpu = max_across_last_axis(device_context, device_queue, x_gpu)

    softmax_val = cl_array.empty_like(x_gpu)

    global_shape_col = x_gpu.shape[-1]
    global_shape_row = np.prod(x_gpu.shape[:-1]).astype(np.int32)
    global_shape = (global_shape_row, global_shape_col)

    event = softmax_gpu_program.softmax(device_queue, global_shape, None,
                                        x_gpu.data, max_val_gpu.data, input_length_gpu.data, softmax_val.data)
    event.wait()

    return softmax_val


def max_across_last_axis(device_context, device_queue, x_gpu):
    cl_array_max_program = cl.Program(device_context, cl_array_max_source_code).build()

    last_axis_length = cl_array.to_device(device_queue, np.array(x_gpu.shape[-1], dtype=np.int32))
    out_gpu = cl_array.empty(device_queue, x_gpu.shape[:-1], dtype=np.float32)

    event = cl_array_max_program.max_across_last_axis(device_queue, (np.prod(out_gpu.shape),), None,
                                                      x_gpu.data, last_axis_length.data, out_gpu.data)
    event.wait()
    return out_gpu


def arg_max_across_last_axis(device_context, device_queue, x_gpu):
    cl_array_max_program = cl.Program(device_context, cl_array_max_source_code).build()

    last_axis_length = cl_array.to_device(device_queue, np.array(x_gpu.shape[-1], dtype=np.int32))
    out_gpu = cl_array.empty(device_queue, x_gpu.shape[:-1], dtype=np.float32)

    event = cl_array_max_program.arg_max_across_last_axis(device_queue, (np.prod(out_gpu.shape),), None,
                                                          x_gpu.data, last_axis_length.data, out_gpu.data)
    event.wait()
    return out_gpu
