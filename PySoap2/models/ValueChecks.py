def as_list_of_data_type(val, data_type, data_name):
    if isinstance(val, data_type):
        return [val]
    elif all([isinstance(v, data_type) for v in val]):
        return val
    raise ValueError(f'{data_name} needs to be an instance of {data_type.__name__}, or a list of'
                     f' {data_type.__name__}')


def check_valid_targets_length(targets_as_list, output_length, data_name):
    if len(targets_as_list) != output_length:
        raise ValueError(f'{data_name} is expecting {output_length} targets, but got {len(targets_as_list)}')


def validate_model(model):
    start_layer, end_layers = model.input_layer, model.output_layers

    if start_layer is None and end_layers is None:
        return True

    """ Checks to see if there is a valid that connects the input layer to the output layer(s) """
    for end_layer in end_layers:
        if _no_valid_path(start_layer, end_layer):
            raise ValueError('No path from the input layer to the output layer')

    if _start_to_end_is_different_as_end_to_start(model):
        raise ValueError('Model has branches that do not connect to the output layer. Either remove'
                         ' these connections or use the Concatenate Layer to combine them.')


def _no_valid_path(start_layer, end_layer):
    if len(start_layer.children) == 0:
        return start_layer != end_layer
    if end_layer in start_layer.children:
        return False
    return all([_no_valid_path(child, end_layer) for child in start_layer.children])


def _start_to_end_is_different_as_end_to_start(model):
    """ Checks if the nodes encountered when starting from the input the the output
        is the same as the nodes encountered when starting from the output to the input
    """
    return len(model.layers_by_number_of_parents) != len(model.layers_by_number_of_children)