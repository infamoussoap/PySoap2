import numpy as np
import pandas as pd


def get_attributes_of_full_model(model):
    """ The attributes of the model, the layers, and the structure of the network """
    full_model_dictionary = {}

    full_model_dictionary.update(get_attributes_of_layers(model))
    full_model_dictionary.update(get_attributes_of_network_tree(model))
    full_model_dictionary.update((get_attributes_of_model(model)))

    return full_model_dictionary


def get_attributes_of_layers(model):
    """
        Parameters
        ----------
        model : :obj:`PySoap2.models.Model`

        Returns
        -------
        dict of str - dict
    """
    layer_attributes = {layer.id: _get_attributes_of_layer(layer)
                        for layer in model.layers_by_number_of_parents}

    return layer_attributes


def _get_attributes_of_layer(layer):
    """ Returns the attributes of the layer as if it was no longer inherited from
        `NetworkNode`, that is, pruned from the tree.

        Notes
        -----
        hdf5 files don't allow the saving on objects, only of np.arrays. Hence
        we remove the parent and children nodes.
    """
    layer_attributes = layer.__dict__.copy()

    del layer_attributes['parents']
    del layer_attributes['children']

    return layer_attributes


def get_attributes_of_network_tree(model):
    """ Returns the network structure of the model as a dictionary """
    parents_adjacency_matrix = _parents_as_weighted_adjacency_matrix(model)
    children_adjacency_matrix = _children_as_weighted_adjacency_matrix(model)

    network_dictionary = {'parents_adjacency_matrix_values': parents_adjacency_matrix.values,
                          'parents_adjacency_matrix_column_names': np.array(list(parents_adjacency_matrix.columns),
                                                                            'S'),
                          'children_adjacency_matrix_values': children_adjacency_matrix.values,
                          'children_adjacency_matrix_column_names': np.array(list(children_adjacency_matrix.columns),
                                                                             'S')}

    return network_dictionary


def _parents_as_weighted_adjacency_matrix(model):
    """ Returns the adjacency matrix of the parent nodes, weighted in the order
        it occurs in the parents tuple. That is,
            0 - No Edge Exists
            1 - First parent in tuple
            2 - Second parent in tuple
            etc.
    """
    layer_ids = [layer.id for layer in model.layers_by_number_of_parents]
    adjacency_matrix = pd.DataFrame(0, index=layer_ids, columns=layer_ids)

    for layer in model.layers_by_number_of_parents:
        for i, parent in enumerate(layer.parents, 1):
            adjacency_matrix.loc[layer.id, parent.id] = i

    return adjacency_matrix


def _children_as_weighted_adjacency_matrix(model):
    """ Returns the adjacency matrix of the children nodes, weighted in the order
        it occurs in the children tuple. That is,
            0 - No Edge Exists
            1 - First child in tuple
            2 - Second child in tuple
            etc.
    """
    layer_ids = [layer.id for layer in model.layers_by_number_of_parents]
    adjacency_matrix = pd.DataFrame(0, index=layer_ids, columns=layer_ids)

    for layer in model.layers_by_number_of_parents:
        for i, child in enumerate(layer.children, 1):
            adjacency_matrix.loc[layer.id, child.id] = i

    return adjacency_matrix


def get_attributes_of_model(model):
    """ These are attributes that are only known the the base model class """
    model_dictionary = {'input_layer_id': model.input_layer.id,
                        'output_layer_id': model.output_layer.id,
                        'optimizer_name': type(model.optimizer).__name__,
                        'optimizer_attributes': model.optimizer.__dict__,
                        'loss_function': model.loss_function,
                        'metric_function': model.metric_function}

    return model_dictionary
