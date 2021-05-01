import pyopencl as cl


def clip_cl_array_by_min_value_in_place(device_context):
    return cl.elementwise.ElementwiseKernel(device_context,
                                            "float *x, float threshold",
                                            "x[i] = x[i] > threshold ? x[i] : threshold",
                                            "clip_in_place_elementwise")


def clip_cl_array_by_max_value_in_place(device_context):
    return cl.elementwise.ElementwiseKernel(device_context,
                                            "float *x, float threshold",
                                            "x[i] = x[i] < threshold ? x[i] : threshold",
                                            "clip_in_place_elementwise")


def clip_cl_array_in_place(device_context, array, min_val, max_val):
    if min_val is not None:
        clip_cl_array_by_min_value_in_place(device_context)(array, min_val)

    if max_val is not None:
        clip_cl_array_by_max_value_in_place(device_context)(array, max_val)
