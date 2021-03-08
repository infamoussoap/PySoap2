def check_built(function):
    def wrapper(inst, *args, **kwargs):
        if not inst.built:
            raise ValueError(f'{type(inst).__name__} Layer must be built before it is used.')
        return function(inst, *args, **kwargs)
    return wrapper
