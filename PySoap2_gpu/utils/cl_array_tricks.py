import numpy as np
import pyopencl as cl

from pyopencl import clmath
from pyopencl.elementwise import ElementwiseKernel
import pyopencl.array as cl_array

from .cl_array_tricks_c_code import *

from PySoap2_gpu.Exceptions import check_for_valid_context


def take(array, indices):
    """ Returns [array[indices[0]], array[indices[1]], ..., array[indices[n]]]

        Parameters
        ----------
        array : cl_array.Array or np.ndarray
        indices : list[int]
    """
    try:
        sub_array = [array[i] for i in indices]
    except TypeError:
        raise TypeError("array must be subscriptable")
    except IndexError as e:
        raise IndexError("indices out of range for array") from e

    if isinstance(sub_array[0], cl_array.Array):
        return cl_array.stack(sub_array, axis=0)
    elif isinstance(array[0], np.ndarray):
        return np.stack(sub_array, axis=0)

    raise ValueError("take only works when array is np.ndarray or cl_array.Array")


class ClArrayTricks:
    initialized = False

    clip_cl_array_by_min_value_in_place = None
    clip_cl_array_by_max_value_in_place = None
    cl_array_max_program = None
    cl_array_sum_program = None
    cl_array_mean_program = None
    cl_array_var_program = None
    cl_array_pad_images_program = None
    flip_across_0_1_axis_program = None
    transpose_last_two_axis_program = None

    device_context = None
    device_queue = None

    def __init__(self, device_context, device_queue):
        # If this class is initialized, it means that the programs is already on the device
        if ClArrayTricks.initialized:
            return

        ClArrayTricks.device_context = device_context
        ClArrayTricks.device_queue = device_queue

        ClArrayTricks.clip_cl_array_by_min_value_in_place = ElementwiseKernel(device_context,
                                                                              "double *x, double threshold",
                                                                              "x[i] = x[i] > threshold ? "
                                                                              "x[i] : threshold",
                                                                              "clip_in_place_elementwise")

        ClArrayTricks.clip_cl_array_by_max_value_in_place = ElementwiseKernel(device_context,
                                                                              "double *x, double threshold",
                                                                              "x[i] = x[i] < threshold ? "
                                                                              "x[i] : threshold",
                                                                              "clip_in_place_elementwise")

        ClArrayTricks.cl_array_max_program = cl.Program(device_context, cl_array_max_source_code).build()
        ClArrayTricks.cl_array_sum_program = cl.Program(device_context, cl_array_sum_across_axis_source_code).build()
        ClArrayTricks.cl_array_mean_program = cl.Program(device_context, mean_across_axis_c_code).build()
        ClArrayTricks.cl_array_var_program = cl.Program(device_context, var_across_axis_c_code).build()
        ClArrayTricks.cl_array_pad_images_program = cl.Program(device_context, pad_images_c_code).build()
        ClArrayTricks.flip_across_0_1_axis_program = cl.Program(device_context, flip_across_0_1_axis_c_code).build()
        ClArrayTricks.transpose_last_two_axis_program = cl.Program(device_context, transpose_last_two_axis_c_code).build()

        ClArrayTricks.initialized = True

    @staticmethod
    def clip_cl_array_in_place(array, min_val, max_val):
        check_for_valid_context(ClArrayTricks.device_context, array)

        if min_val is not None:
            ClArrayTricks.clip_cl_array_by_min_value_in_place(array, min_val)

        if max_val is not None:
            ClArrayTricks.clip_cl_array_by_max_value_in_place(array, max_val)

    @staticmethod
    def max_across_last_axis(x_gpu):
        check_for_valid_context(ClArrayTricks.device_context, x_gpu)

        last_axis_length = np.int32(x_gpu.shape[-1])
        out_gpu = cl_array.empty(ClArrayTricks.device_queue, x_gpu.shape[:-1], dtype=np.float64)

        event = ClArrayTricks.cl_array_max_program.max_across_last_axis(ClArrayTricks.device_queue,
                                                                        (np.prod(out_gpu.shape),), None,
                                                                        x_gpu.data, last_axis_length, out_gpu.data)
        event.wait()

        return out_gpu

    @staticmethod
    def arg_max_across_last_axis(x_gpu):
        check_for_valid_context(ClArrayTricks.device_context, x_gpu)

        last_axis_length = np.int32(x_gpu.shape[-1])
        out_gpu = cl_array.empty(ClArrayTricks.device_queue, x_gpu.shape[:-1], dtype=np.int32)

        event = ClArrayTricks.cl_array_max_program.arg_max_across_last_axis(ClArrayTricks.device_queue,
                                                                            (np.prod(out_gpu.shape),), None,
                                                                            x_gpu.data, last_axis_length,
                                                                            out_gpu.data)
        event.wait()

        return out_gpu

    @staticmethod
    def sum_across_0_axis(array):
        """ array assumed to be a cl_array, not a list of cl_arrays.

            If you want to sum a list of cl_arrays just use the reduce method
        """
        check_for_valid_context(ClArrayTricks.device_context, array)

        N, *input_shape = array.shape

        input_shape = tuple(input_shape)
        input_length = np.int32(np.prod(input_shape))
        N = np.int32(N)

        out = cl_array.empty(ClArrayTricks.device_queue, input_shape, dtype=np.float64)

        event = ClArrayTricks.cl_array_sum_program.sum_across_0_axis(ClArrayTricks.device_queue, (input_length,), None,
                                                                     array.data, input_length, N, out.data)
        event.wait()

        return out

    @staticmethod
    def mean_across_0_axis(x_val_device):
        check_for_valid_context(ClArrayTricks.device_context, x_val_device)

        queue = ClArrayTricks.device_queue
        mean_program = ClArrayTricks.cl_array_mean_program

        N, *input_shape = x_val_device.shape

        input_shape = tuple(input_shape)
        input_length = np.int32(np.prod(input_shape))
        N = np.int32(N)

        out = cl_array.empty(queue, input_shape, dtype=np.float64)

        event = mean_program.mean_across_0_axis(queue, (input_length,), None,
                                                x_val_device.data, input_length, N, out.data)
        event.wait()

        return out

    @staticmethod
    def var_across_0_axis(x_val_device):
        check_for_valid_context(ClArrayTricks.device_context, x_val_device)

        queue = ClArrayTricks.device_queue
        var_program = ClArrayTricks.cl_array_var_program

        N, *input_shape = x_val_device.shape

        input_shape = tuple(input_shape)
        input_length = np.int32(np.prod(input_shape))
        N = np.int32(N)

        out = cl_array.empty(queue, input_shape, dtype=np.float64)

        mean = ClArrayTricks.mean_across_0_axis(x_val_device)

        event = var_program.var_across_0_axis(queue, (input_length,), None,
                                              x_val_device.data, mean.data, input_length, N, out.data)
        event.wait()

        return out

    @staticmethod
    def std_across_0_axis(x_val_device):
        check_for_valid_context(ClArrayTricks.device_context, x_val_device)
        return clmath.sqrt(ClArrayTricks.var_across_0_axis(x_val_device))

    @staticmethod
    def pad_images(images, upper_pad, lower_pad, left_pad, right_pad):
        """ images assumed to be a series of 3d images, and padding occurs on the width and height dimensions

            Parameters
            ----------
            images : (n, i, j, k) cl_array
            upper_pad : int
            lower_pad : int
            left_pad : int
            right_pad : int
        """
        check_for_valid_context(ClArrayTricks.device_context, images)

        queue = ClArrayTricks.device_queue
        pad_images_program = ClArrayTricks.cl_array_pad_images_program

        N, *image_shape = images.shape
        padded_image_shape = (image_shape[0] + upper_pad + lower_pad,
                              image_shape[1] + left_pad + right_pad,
                              image_shape[2])
        padded_images = cl_array.zeros(queue, (N, *padded_image_shape), np.float64)

        image_height, image_width, image_channels = [np.int32(x) for x in image_shape]
        padded_image_height, padded_image_width = np.int32(padded_image_shape[0]), np.int32(padded_image_shape[1])
        row_start_index, column_start_index = np.int32(upper_pad), np.int32(left_pad)

        global_shape = (int(np.prod(images.shape)),)
        event = pad_images_program.pad_images(queue, global_shape, None,
                                              images.data,
                                              image_height, image_width, image_channels,
                                              padded_image_height, padded_image_width,
                                              row_start_index, column_start_index,
                                              padded_images.data)
        event.wait()

        return padded_images

    @staticmethod
    def flip_across_0_1_axis(x):
        check_for_valid_context(ClArrayTricks.device_context, x)

        queue = ClArrayTricks.device_queue
        program = ClArrayTricks.flip_across_0_1_axis_program

        out = cl_array.zeros_like(x)
        global_shape = (np.int32(x.shape[0]), np.int32(x.shape[1]), np.int32(np.prod(x.shape[2:])))
        event = program.flip_across_0_1_axis(queue, global_shape, None,
                                             x.data, *global_shape, out.data)
        event.wait()

        return out

    @staticmethod
    def transpose_last_two_axis(x):
        check_for_valid_context(ClArrayTricks.device_context, x)

        queue = ClArrayTricks.device_queue
        program = ClArrayTricks.transpose_last_two_axis_program

        output_shape = (*x.shape[:-2], x.shape[-1], x.shape[-2])
        out = cl_array.zeros(queue, output_shape, np.float64)

        global_shape = (int(np.prod(x.shape[:-2])), x.shape[-2], x.shape[-1])
        event = program.transpose_last_two_axis(queue, global_shape, None,
                                               x.data, np.int32(x.shape[-2]), np.int32(x.shape[-1]), out.data)
        event.wait()

        return out
