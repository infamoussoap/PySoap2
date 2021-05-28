def simplify_recursive_dict(working_dict, delimiter='/'):
    """ Given a recursive dictionary (a dictionary of dictionary of dictionary ...) this
        will simplify the dictionary so it is simply a dictionary of str - :obj: pair.
        Keys of dictionaries are separated by the delimiter
    """

    new_dict = {}
    for (key, val) in working_dict.items():
        if isinstance(val, dict):
            if len(val) == 0:
                new_dict.update({key: None})
            else:
                temp_dict = simplify_recursive_dict(val)
                new_dict.update({f'{key}{delimiter}{new_key}': new_val
                                 for (new_key, new_val) in temp_dict.items()})
        else:
            new_dict.update({key: val})

    return new_dict


def unpack_to_recursive_dict(working_dict, delimiter='/'):
    """ Given a non-recursive dictionary of str - :obj: pair, this will unpack it to a dictionary of
        dictionaries
    """
    new_dict = {}
    for (key, val) in working_dict.items():
        if '/' in key:
            higher_key, lower_key = key.split(delimiter, 1)
            if higher_key in new_dict:
                new_dict[higher_key].update(unpack_to_recursive_dict({lower_key: val}))
            else:
                new_dict.update({higher_key: unpack_to_recursive_dict({lower_key: val})})
        else:
            new_dict.update({key: val})
    return new_dict
