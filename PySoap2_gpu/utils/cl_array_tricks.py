import pyopencl as cl

platform = cl.get_platforms()[0]
device = platform.get_devices()[0]

device_context = cl.Context([device])
device_queue = cl.CommandQueue(device_context)

clip_cl_array_by_min_value_in_place = cl.elementwise.ElementwiseKernel(device_context,
                                                                       "float *x, float threshold",
                                                                       "x[i] = x[i] > threshold ? x[i] : threshold",
                                                                       "clip_in_place_elementwise")

clip_cl_array_by_max_value_in_place = cl.elementwise.ElementwiseKernel(device_context,
                                                                       "float *x, float threshold",
                                                                       "x[i] = x[i] < threshold ? x[i] : threshold",
                                                                       "clip_in_place_elementwise")


def clip_cl_array_in_place(array, min_val, max_val):
    if min_val is not None:
        clip_cl_array_by_min_value_in_place(array, min_val)

    if max_val is not None:
        clip_cl_array_by_max_value_in_place(array, max_val)
