import numpy as np
import pandas as pd

import inspect


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
    layer_attributes = {layer.id: get_layer_attributes_as_dict(layer)
                        for layer in model.layers_by_number_of_parents}

    return layer_attributes


def get_layer_attributes_as_dict(layer):
    layer_attributes = get_init_attributes_of_cls_as_dict(layer)
    weight_attributes = layer.get_weights(as_dict=True)

    layer_attributes.update(weight_attributes)

    return layer_attributes


def get_init_attributes_of_cls_as_dict(cls):
    attributes = get_init_attributes_names_of_cls(cls)
    cls_dict = cls.__dict__

    return {p.name: cls_dict[p.name] for p in attributes}


def get_init_attributes_names_of_cls(cls):
    init_signature = inspect.signature(cls.__init__)
    attributes = [p for p in init_signature.parameters.values()
                  if p.name != 'self' and p.kind != p.VAR_KEYWORD and p.kind != p.VAR_POSITIONAL]

    return attributes


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
                        'optimizer_attributes': get_optimizer_attributes_as_dict(model.optimizer),
                        'loss_function': model.loss_function,
                        'metric_function': model.metric_function}

    return model_dictionary


def get_optimizer_attributes_as_dict(optimizer):
    optimizer_attributes = get_init_attributes_of_cls_as_dict(optimizer)
    parameters_attributes = optimizer.parameters_()

    optimizer_attributes.update(parameters_attributes)

    return optimizer_attributes
