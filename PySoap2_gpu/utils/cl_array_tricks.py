import numpy as np
import pyopencl as cl

from pyopencl import clmath
from pyopencl.elementwise import ElementwiseKernel
import pyopencl.array as cl_array

from .cl_math_functions_c_code import cl_array_max_source_code
from .cl_math_functions_c_code import cl_array_sum_across_axis_source_code
from .cl_math_functions_c_code import mean_across_axis_c_code
from .cl_math_functions_c_code import var_across_axis_c_code


class ClArrayTricks:
    initialized = False

    clip_cl_array_by_min_value_in_place = None
    clip_cl_array_by_max_value_in_place = None
    cl_array_max_program = None
    cl_array_sum_program = None
    cl_array_mean_program = None
    cl_array_var_program = None

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
        ClArrayTricks.cl_array_sum_program = cl.Program(device_context, cl_array_sum_across_axis_source_code).build()
        ClArrayTricks.cl_array_mean_program = cl.Program(device_context, mean_across_axis_c_code).build()
        ClArrayTricks.cl_array_var_program = cl.Program(device_context, var_across_axis_c_code).build()

        ClArrayTricks.initialized = True

    @staticmethod
    def clip_cl_array_in_place(array, min_val, max_val):
        if min_val is not None:
            ClArrayTricks.clip_cl_array_by_min_value_in_place(array, min_val)

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

    @staticmethod
    def sum_across_0_axis(array):
        """ array assumed to be a cl_array, not a list of cl_arrays.

            If you want to sum a list of cl_arrays just use the reduce method
        """
        N, *input_shape = array.shape

        input_shape = tuple(input_shape)
        input_length = int(np.prod(input_shape))

        input_length_device = cl_array.to_device(ClArrayTricks.device_queue, np.array(input_length, dtype=np.int32))
        N_device = cl_array.to_device(ClArrayTricks.device_queue, np.array(N, dtype=np.int32))
        out = cl_array.empty(ClArrayTricks.device_queue, input_shape, dtype=np.float32)

        event = ClArrayTricks.cl_array_sum_program.sum_across_0_axis(ClArrayTricks.device_queue, (input_length,), None,
                                                                     array.data, input_length_device.data,
                                                                     N_device.data, out.data)
        event.wait()

        return out

    @staticmethod
    def mean_across_0_axis(x_val_device):
        queue = ClArrayTricks.device_queue
        mean_program = ClArrayTricks.cl_array_mean_program

        N, *input_shape = x_val_device.shape

        input_shape = tuple(input_shape)
        input_length = int(np.prod(input_shape))

        input_length_device = cl_array.to_device(queue, np.array(input_length, dtype=np.int32))
        N_device = cl_array.to_device(queue, np.array(N, dtype=np.int32))
        out = cl_array.empty(queue, input_shape, dtype=np.float32)

        event = mean_program.mean_across_0_axis(queue, (input_length,), None,
                                                x_val_device.data, input_length_device.data, N_device.data, out.data)
        event.wait()

        return out

    @staticmethod
    def var_across_0_axis(x_val_device):
        queue = ClArrayTricks.device_queue
        var_program = ClArrayTricks.cl_array_var_program

        N, *input_shape = x_val_device.shape

        input_shape = tuple(input_shape)
        input_length = int(np.prod(input_shape))

        input_length_device = cl_array.to_device(queue, np.array(input_length, dtype=np.int32))
        N_device = cl_array.to_device(queue, np.array(N, dtype=np.int32))
        out = cl_array.empty(queue, input_shape, dtype=np.float32)

        mean = ClArrayTricks.mean_across_0_axis(x_val_device)

        event = var_program.var_across_0_axis(queue, (input_length,), None,
                                              x_val_device.data, mean.data, input_length_device.data,
                                              N_device.data, out.data)
        event.wait()

        return out

    @staticmethod
    def std_across_0_axis(x_val_device):
        return clmath.sqrt(ClArrayTricks.var_across_0_axis(x_val_device))
