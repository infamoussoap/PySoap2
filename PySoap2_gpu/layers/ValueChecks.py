import pyopencl.array as cl_array


def assert_instance_of_cl_array(array):
    if not isinstance(array, cl_array.Array):
        raise ValueError(f'Arrays must be an instance of pyopencl.array.Array not {type(array)}')
