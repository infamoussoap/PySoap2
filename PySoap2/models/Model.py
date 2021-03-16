import numpy as np
import pandas as pd
import functools
import random

import h5py

from PySoap2.optimizers import Optimizer, get_optimizer
from PySoap2 import get_error_function, get_metric_function

from .dictionary_tricks import simplify_recursive_dict, unpack_to_recursive_dict


class Model:
    @staticmethod
    def _is_valid_model(start_layer, end_layer):
        if start_layer is None and end_layer is None:
            return True

        """ Checks to see if there is a valid that connects the input layer to the output layer """
        if len(start_layer.children) == 0:
            return start_layer == end_layer
        if end_layer in start_layer.children:
            return True

        return any([Model._is_valid_model(child, end_layer) for child in start_layer.children])

    def __init__(self, input_layer, output_layer):
        if not self._is_valid_model(input_layer, output_layer):
            raise ValueError('There is no path from the input layer to the output layer.')

        self.input_layer = input_layer
        self.output_layer = output_layer

        self.optimizer = None

        self.loss_function = None
        self.metric_function = None

    def build(self, loss_function, optimizer, metrics=None):
        """ Build the layers in the tree network, and save attributes of Model """
        if isinstance(optimizer, Optimizer):
            self.optimizer = optimizer
        elif isinstance(optimizer, str):
            self.optimizer = get_optimizer(optimizer)
        else:
            raise ValueError("optimizer must be an instance of Optimizer or str")

        self.loss_function = loss_function
        self.metric_function = metrics

        for layer in self.layers_by_number_of_parents:
            layer.build()

    @property
    @functools.lru_cache()
    def layers_by_number_of_parents(self):
        """ Returns a list by order of the nodes with the least amount of parents to the most parents """
        current_layers = [self.output_layer]  # Terminal node will have the most parents
        layer_order = []

        while len(current_layers) > 0:
            for layer in current_layers:
                if layer in layer_order:
                    layer_order.remove(layer)
                layer_order.append(layer)

            parents_of_current_layers = [layer.parents for layer in current_layers]
            current_layers = set(functools.reduce(lambda x, y: x + y, parents_of_current_layers))

        return layer_order[::-1]

    def predict(self, z, output_only=True):
        """ Perform forward propagation of the whole network

            Parameters
            ----------
            z : (N, *input_shape) np.array
                The Input
            output_only : bool
                If true then only the model output will be returned
                Otherwise the pre and post activations will be returned as a dictionary

            Returns
            -------
            (N, *output_shape) np.array
                This is returned in output_only=True

            or

            dict of str - (np.array, np.array)
                This is returned in output_only=False
                The str keys are the layers memory location, i.e. their unique identifier. The associated value
                will be a tuple of the pre and post activations.
        """

        # Input is a special case
        cached_outputs = {self.input_layer.memory_location: self.input_layer.predict(z, output_only=output_only)}

        for layer in self.layers_by_number_of_parents[1:]:  # Input is assumed to have the least number of parents
            layer_id = layer.memory_location

            layer_arg = self._get_layer_predict_arguments(layer, cached_outputs, output_only=output_only)
            cached_outputs[layer_id] = layer.predict(layer_arg, output_only=output_only)

        if not output_only:
            return cached_outputs
        return cached_outputs[self.output_layer.memory_location]

    @staticmethod
    def _get_layer_predict_arguments(layer, cached_outputs, output_only=True):
        """ Returns the arguments to be passed into layer to be predicted

            Parameters
            ----------
            layer : :obj:Layer
                The layer to return the output of
            cached_outputs : dict of str - :obj:
                Stores the outputs of the parent nodes. Note that when calling this layer
                it is assumed that the the root node (or terminal nodes) is inside cached_outputs
            output_only : bool
                If true then cached_outputs is dict of str - np.array
                If false then cached_outputs is dict of str - (np.array, np.array)
        """
        layer_args = [cached_outputs[parent.memory_location] for parent in layer.parents]

        if not output_only:
            layer_args = [post_activation for (pre_activation, post_activation) in layer_args]

        if len(layer.parents) == 1:
            return layer_args[0]
        return layer_args

    def evaluate(self, x_test, y_test):
        """ Return the MSE of the model prediction
            Parameters
            ----------
            x_test : np.array
                X_test is assumed to be a list of all the inputs to be forward propagated. In particular
                it is assumed that the first index of X_test is the index that inputs is accessed by
            y_test : np.array
                y_test is the associated list of outputs to the list of inputs X_test.
            Returns
            -------
            str
                The error
        """
        prediction = self.predict(x_test)

        loss_val = self._loss_function(prediction, y_test)

        eval_str = f'{self.loss_function}: {format(loss_val, ".4f")}'

        if self.metric_function is not None:
            metric_val = self._metric(prediction, y_test)
            eval_str += f' - {self.metric_function}: {format(metric_val, ".4f")}'

        return eval_str

    def train(self, x_train, y_train, epochs=100, batch_size=None, verbose=True):
        """ Train the neural network by means of back propagation
            Parameters
            ----------
            x_train : np.array
                x_train is assumed to be a list of all the inputs to be forward propagated. In particular
                it is assumed that the first index of x_train is the index that inputs is accessed by
            y_train : np.array
                y_train is the associated list of outputs to the list of inputs x_train. More specifically,
                the neural network will be trained to find the map x_train -> y_train
            epochs : :obj:`int`, optional
                Number of times the neural network will see the entire dataset
            batch_size : :obj:`int`, optional
                The batch size for gradient descent. If not defined then `batch_size` is set to the
                length of the dataset, i.e. traditional gradient descent.
            verbose : bool, optional
                If set to `True` then the model performance will be printed after each epoch
        """

        training_length = len(x_train)
        if batch_size is None:
            batch_size = training_length
        index = list(range(training_length))

        for _ in range(epochs):
            if verbose:
                print(f'Training on {len(x_train)} samples')

            random.shuffle(index)
            for i in range(np.ceil(training_length / batch_size).astype(int)):
                start, end = i * batch_size, (i + 1) * batch_size
                batch_x, batch_y = x_train[index[start:end]], y_train[index[start:end]]

                self._back_prop(batch_x, batch_y)

            if verbose:
                start, end = 0, batch_size
                batch_x, batch_y = x_train[index[start:end]], y_train[index[start:end]]
                evaluation = self.evaluate(batch_x, batch_y)
                print(f'Epoch {_ + 1}/{epochs}')
                print(evaluation)

    def _back_prop(self, x_train, y_train):
        """ Perform one iteration of backpropagation on the given batches """

        layer_gradients = self._get_layer_gradients(x_train, y_train)
        parameter_updates = self.optimizer.step(simplify_recursive_dict(layer_gradients))
        parameter_updates_by_layer = unpack_to_recursive_dict(parameter_updates)

        for layer in self.layers_by_number_of_parents[1:]:
            if layer.memory_location in parameter_updates_by_layer:
                layer.update_parameters_(parameter_updates_by_layer[layer.memory_location])

    def _get_layer_gradients(self, x_train, y_train):
        """ Returns the gradients for each layer as a dictionary """
        prediction = self.predict(x_train, output_only=False)

        cached_pre_activation = {key: val[0] for (key, val) in prediction.items()}
        cached_output = {key: val[1] for (key, val) in prediction.items()}
        cached_delta = self._get_layer_deltas(y_train, cached_pre_activation, cached_output)

        grad_dict = {}
        for layer in self.layers_by_number_of_parents[1:]:
            z = tuple([cached_output[parent.memory_location] for parent in layer.parents])
            if len(layer.parents) == 1:
                z = z[0]

            grad_dict[layer.memory_location] = layer.get_parameter_gradients_(cached_delta[layer.memory_location], z)

        return grad_dict

    def _get_layer_deltas(self, y_train, cached_pre_activation, cached_output):
        """ Returns the delta^k for all the layers """

        cached_delta = {}

        output_id = self.output_layer.memory_location

        cached_delta[output_id] = self._loss_function(cached_output[output_id], y_train,
                                                      grad=True)  # Gradient of output
        if self.loss_function == 'cross_entropy':
            unique_identifier = self.output_layer.memory_location
            cached_delta[unique_identifier] = cached_output[unique_identifier] - y_train

        for layer in self.layers_by_number_of_children:
            for parent in layer.parents:
                delta = tuple([cached_delta[child.memory_location] for child in parent.children])
                if len(parent.children) == 1:
                    delta = delta[0]

                g_prime = parent.activation_function_(cached_pre_activation[parent.memory_location], grad=True)
                z = cached_output[parent.memory_location]

                cached_delta[parent.memory_location] = layer.get_delta_backprop_(g_prime, delta, z)

        return cached_delta

    @property
    @functools.lru_cache()
    def layers_by_number_of_children(self):
        """ Returns a list by order of the nodes with the least amount of children to the most children """
        current_layers = [self.input_layer]  # Input node will have the most children
        layer_order = []

        while len(current_layers) > 0:
            for layer in current_layers:
                if layer in layer_order:
                    layer_order.remove(layer)
                layer_order.append(layer)

            children_of_current_layers = [layer.children for layer in current_layers]
            current_layers = set(functools.reduce(lambda x, y: x + y, children_of_current_layers))

        return layer_order[::-1]

    @property
    def _loss_function(self):
        return get_error_function(self.loss_function)

    @property
    def _metric(self):
        if self.metric_function is not None:
            return get_metric_function(self.metric_function)
        return None

    def save_model(self, file_path):
        """ Save the self into file_path

            Notes
            -----
            It is assumed that file_path is a .hdf file
        """
        parents_adjacency_matrix = self._parents_as_weighted_adjacency_matrix
        children_adjacency_matrix = self._children_as_weighted_adjacency_matrix

        layer_attributes = {layer.id: self.pruned_layer_attributes(layer)
                            for layer in self.layers_by_number_of_parents}

        model_dictionary = layer_attributes.copy()
        model_dictionary['parents_adjacency_matrix_values'] = parents_adjacency_matrix.values
        model_dictionary['parents_adjacency_matrix_column_names'] = np.array(list(parents_adjacency_matrix.columns),
                                                                             'S')

        model_dictionary['children_adjacency_matrix_values'] = children_adjacency_matrix.values
        model_dictionary['children_adjacency_matrix_column_names'] = np.array(list(children_adjacency_matrix.columns),
                                                                              'S')

        model_dictionary['input_layer_id'] = self.input_layer.id
        model_dictionary['output_layer_id'] = self.output_layer.id

        simplified_model_dictionary = simplify_recursive_dict(model_dictionary)

        with h5py.File(file_path, 'w') as h:
            for key, val in simplified_model_dictionary.items():
                h.create_dataset(key, data=val)

    @property
    def _parents_as_weighted_adjacency_matrix(self):
        """ Returns the adjacency matrix of the parent nodes, weighted in the order
            it occurs in the parents tuple. That is,
                0 - No Edge Exists
                1 - First parent in tuple
                2 - Second parent in tuple
                etc.
        """
        layer_ids = [layer.id for layer in self.layers_by_number_of_parents]
        adjacency_matrix = pd.DataFrame(0, index=layer_ids, columns=layer_ids)

        for layer in self.layers_by_number_of_parents:
            for i, parent in enumerate(layer.parents, 1):
                adjacency_matrix.loc[layer.id, parent.id] = i

        return adjacency_matrix

    @property
    def _children_as_weighted_adjacency_matrix(self):
        """ Returns the adjacency matrix of the children nodes, weighted in the order
            it occurs in the children tuple. That is,
                0 - No Edge Exists
                1 - First child in tuple
                2 - Second child in tuple
                etc.
        """
        layer_ids = [layer.id for layer in self.layers_by_number_of_parents]
        adjacency_matrix = pd.DataFrame(0, index=layer_ids, columns=layer_ids)

        for layer in self.layers_by_number_of_parents:
            for i, child in enumerate(layer.children, 1):
                adjacency_matrix.loc[layer.id, child.id] = i

        return adjacency_matrix

    @staticmethod
    def pruned_layer_attributes(layer):
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
