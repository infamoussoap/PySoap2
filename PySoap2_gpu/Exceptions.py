class InvalidContextError(Exception):
    pass


def check_for_valid_context(context, *args):
    if any([arg.context != context for arg in args]):
        raise InvalidContextError("Input has incompatible context")
