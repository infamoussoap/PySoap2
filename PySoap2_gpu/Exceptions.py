import pyopencl.array as cl_array


class InvalidContextError(Exception):
    pass


def check_for_valid_context(context, *args):
    try:
        if any([arg.context != context for arg in args]):
            raise InvalidContextError("Input has incompatible context")
    except AttributeError:
        invalid_args = [type(arg).__name__ for arg in args if not isinstance(arg, cl_array.Array)]
        raise ValueError(f"Was expecting instance of cl_array.Array, instead got {set(invalid_args)}")
