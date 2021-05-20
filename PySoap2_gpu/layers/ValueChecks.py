import pyopencl.array as cl_array


def assert_instance_of_cl_array(array):
    if not isinstance(array, cl_array.Array):
        raise ValueError(f'Arrays must be an instance of pyopencl.array.Array not {type(array)}')


def check_built(function):
    def wrapper(inst, *args, **kwargs):
        if not inst.built:
            raise ValueError(f'{type(inst).__name__} Layer must be built before it is used.')
        return function(inst, *args, **kwargs)
    return wrapper
