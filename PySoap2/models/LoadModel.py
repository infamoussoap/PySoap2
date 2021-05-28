import pandas as pd
import h5py
from inspect import signature

import PySoap2


def load_model(file_path):
    with h5py.File(file_path, 'r') as h:
        model_attributes = unpack_recursive_hdf5(h)

    instances_of_layer_id = get_instances_of_layer(model_attributes)

    update_children_of_layers(instances_of_layer_id, model_attributes)
    update_parents_of_layers(instances_of_layer_id, model_attributes)

    input_layer = instances_of_layer_id[model_attributes['input_layer_id']]
    output_layer = instances_of_layer_id[model_attributes['output_layer_id']]

    model = PySoap2.models.Model(input_layer, output_layer)

    loss_function = model_attributes['loss_function']
    metric_function = model_attributes['metric_function'] if 'metric_function' in model_attributes else None
    optimizer = model_attributes['optimizer_name']

    # Model building randomly initializes the weights of the layers
    # Need to reset them to the loaded weights
    model.build(loss_function=loss_function, optimizer=optimizer, metrics=metric_function)

    model.optimizer.__dict__.update(model_attributes['optimizer_attributes'])

    # Update the layer attributes to the saved attributes
    for layer_id, layer in instances_of_layer_id.items():
        layer_attributes = model_attributes[layer_id]
        if layer_attributes != b'null':
            pass
            layer.__dict__.update(layer_attributes)

    return model


def get_instances_of_layer(model_attributes):
    """ Returns a dictionary with keys of the layer_id and values as the instances
        of the layers with their correct attributes.

        Notes
        -----
        parents and children are not in model_attributes, and will need to be added
    """
    instances_of_layer_id = {}
    for layer_id, layer_attributes in model_attributes.items():
        if is_valid_layer_id(layer_id):
            # The metaclass to that is used to create the layer instance
            layer_metaclass = get_metaclass_from_id(layer_id)
            layer_attributes = model_attributes[layer_id]

            # Create a new instance of the metaclass and update the attributes to what is given
            # in the dictionary
            instances_of_layer_id[layer_id] = load_layer(layer_metaclass, layer_attributes)

    return instances_of_layer_id


def update_children_of_layers(instances_of_layer_id, model_attributes):
    """ Updates the children of the layer instances. The children, and their order, are determined by
        the children adjacency matrix given by model_attributes
    """
    children_adjacency_matrix = get_adjacency_matrix_of_children(model_attributes)

    for current_layer_id, current_layer_instance in instances_of_layer_id.items():
        children_links_of_current_layer = children_adjacency_matrix.loc[current_layer_id, :]
        children_id_of_current_layer = children_links_of_current_layer[children_links_of_current_layer > 0]

        # Weight of links are the order they come in, in the children tuple
        children_id_of_current_layer = children_id_of_current_layer.sort_values()
        children_instances = tuple([instances_of_layer_id[layer_id] for layer_id in children_id_of_current_layer.index])

        current_layer_instance.children = children_instances


def update_parents_of_layers(instances_of_layer_id, model_attributes):
    """ Updates the parents of the layer instances. The parents, and their order, are determined by
        the parents adjacency matrix given by model_attributes
    """
    parent_adjacency_matrix = get_adjacency_matrix_of_parents(model_attributes)

    for current_layer_id, current_layer_instance in instances_of_layer_id.items():
        parent_links_of_current_layer = parent_adjacency_matrix.loc[current_layer_id, :]
        parent_id_of_current_layer = parent_links_of_current_layer[parent_links_of_current_layer > 0]

        # Weight of links are the order they come in, in the parent tuple
        parent_id_of_current_layer = parent_id_of_current_layer.sort_values()
        parent_instances = tuple([instances_of_layer_id[layer_id] for layer_id in parent_id_of_current_layer.index])

        current_layer_instance.parents = parent_instances


def load_layer(layer_metaclass, layer_attributes):
    """ Returns an instance of layer_metaclass with the given attributes

        Parameters
        ----------
        layer_metaclass: :obj:abc.ABCMeta
            The metaclass whoes instance is the desired class
        layer_attributes : dict of str - :obj:
            The attributes of layer_name
    """
    init_kwargs = get_layer_init_kw_arguments(layer_metaclass, layer_attributes)

    layer_instance = layer_metaclass(**init_kwargs)
    # layer_instance.__dict__.update(layer_attributes)
    return layer_instance


def get_layer_init_kw_arguments(layer_metaclass, layer_attributes):
    """ Returns the arguments required to initialise a new instance of the layer

        Parameters
        ----------
        layer_metaclass: :obj:abc.ABCMeta
            The metaclass whoes instance is the desired class
        layer_attributes : dict of str - :obj:
            The attributes of layer_name
    """

    sig = signature(layer_metaclass).parameters
    kwargs = {key: layer_attributes[key] for key in sig.keys() if key in layer_attributes}
    return kwargs


def get_metaclass_from_id(layer_id):
    """ Returns the metaclass of the layer_id

        Parameters
        ----------
        layer_id : str
            The name of the metaclass is assumed to be the
    """
    key_words = layer_id.split('_')[:-1]
    layer_type_name = '_'.join(key_words)

    return PySoap2.layers.__dict__[layer_type_name]


def is_valid_layer_id(layer_id):
    """ layer_id can't be the Parent of Concatenate, nor the children of the split node

        Notes
        -----
        Perhaps using a regex to make sure it follows the correct format as well
    """

    key_words = layer_id.split('_')[:-1]
    layer_type_name = '_'.join(key_words)

    return layer_type_name in PySoap2.layers.__dict__.keys()


def get_adjacency_matrix_of_children(model_attributes):
    return get_adjacency_matrix(model_attributes, 'children')


def get_adjacency_matrix_of_parents(model_attributes):
    return get_adjacency_matrix(model_attributes, 'parents')


def get_adjacency_matrix(model_attributes, type_):
    column_names_as_bytes = model_attributes[f'{type_}_adjacency_matrix_column_names']
    column_names = [x.decode("utf-8") for x in column_names_as_bytes]

    adjacency_matrix = pd.DataFrame(0, index=column_names, columns=column_names)

    adjacency_matrix.iloc[:, :] = model_attributes[f'{type_}_adjacency_matrix_values']

    return adjacency_matrix


def unpack_recursive_hdf5(file):
    unpacked = {}
    for key in file.keys():
        if isinstance(file[key], h5py._hl.group.Group):
            unpacked.update({key: unpack_recursive_hdf5(file[key])})
        else:
            unpacked.update({key: file[key][()]})
    return unpacked


def check_null(val):
    if isinstance(val, str):
        return val == 'null'
    return False
