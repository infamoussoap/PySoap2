def visualize_model(model):
    """ Visualize model as a tree

        Notes
        -----
        There is a lot of dependencies to get graphviz to work. Because visualisation of the model isn't
        necessary, importing graphviz is performed when the function is ran, instead of on runtime.
    """

    from graphviz import Digraph

    dot = Digraph(comment='The Model')
    for i, layer in enumerate(model.layers_by_number_of_parents):
        if type(layer).__name__ == 'Dense':
            name = f'Dense {layer.hidden_nodes}'
        else:
            name = type(layer).__name__

        dot.node(layer.id, name)

    for layer in model.layers_by_number_of_parents:
        for parent in layer.parents:
            dot.edge(parent.id, layer.id)

    return dot
