import numpy as np
import pyopencl as cl

from pyopencl.elementwise import ElementwiseKernel
import pyopencl.array as cl_array

from .cl_math_functions_c_code import cl_array_max_source_code


class ClArrayTricks:
    initialized = False

    clip_cl_array_by_min_value_in_place = None
    clip_cl_array_by_max_value_in_place = None
    cl_array_max_program = None

    device_context = None
    device_queue = None

    def __init__(self, device_context, device_queue):
        ClArrayTricks.device_context = device_context
        ClArrayTricks.device_queue = device_queue

        ClArrayTricks.clip_cl_array_by_min_value_in_place = ElementwiseKernel(device_context,
                                                                              "float *x, float threshold",
                                                                              "x[i] = x[i] > threshold ? "
                                                                              "x[i] : threshold",
                                                                              "clip_in_place_elementwise")

        ClArrayTricks.clip_cl_array_by_max_value_in_place = ElementwiseKernel(device_context,
                                                                              "float *x, float threshold",
                                                                              "x[i] = x[i] < threshold ? "
                                                                              "x[i] : threshold",
                                                                              "clip_in_place_elementwise")

        ClArrayTricks.cl_array_max_program = cl.Program(device_context, cl_array_max_source_code).build()

        ClArrayTricks.initialized = True

    @staticmethod
    def clip_cl_array_in_place(array, min_val, max_val):
        if min_val is not None:
            ClArrayTricks.clip_cl_array_by_max_value_in_place(array, min_val)

        if max_val is not None:
            ClArrayTricks.clip_cl_array_by_max_value_in_place(array, max_val)

    @staticmethod
    def max_across_last_axis(x_gpu):
        last_axis_length = cl_array.to_device(ClArrayTricks.device_queue, np.array(x_gpu.shape[-1], dtype=np.int32))
        out_gpu = cl_array.empty(ClArrayTricks.device_queue, x_gpu.shape[:-1], dtype=np.float32)

        event = ClArrayTricks.cl_array_max_program.max_across_last_axis(ClArrayTricks.device_queue,
                                                                        (np.prod(out_gpu.shape),), None,
                                                                        x_gpu.data, last_axis_length.data, out_gpu.data)
        event.wait()

        return out_gpu

    @staticmethod
    def arg_max_across_last_axis(x_gpu):
        last_axis_length = cl_array.to_device(ClArrayTricks.device_queue, np.array(x_gpu.shape[-1], dtype=np.int32))
        out_gpu = cl_array.empty(ClArrayTricks.device_queue, x_gpu.shape[:-1], dtype=np.int32)

        event = ClArrayTricks.cl_array_max_program.arg_max_across_last_axis(ClArrayTricks.device_queue,
                                                                            (np.prod(out_gpu.shape),), None,
                                                                            x_gpu.data, last_axis_length.data,
                                                                            out_gpu.data)
        event.wait()

        return out_gpu
