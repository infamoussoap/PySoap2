import numpy as np
import pandas as pd
import functools
import random

import h5py

from PySoap2.optimizers import Optimizer, get_optimizer
from PySoap2 import get_error_function, get_metric_function

from .dictionary_tricks import simplify_recursive_dict, unpack_to_recursive_dict
from .SaveModel import get_attributes_of_full_model


def _validate_model(model):
    start_layer, end_layer = model.input_layer, model.output_layer

    if start_layer is None and end_layer is None:
        return True

    """ Checks to see if there is a valid that connects the input layer to the output layer """
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


class Model:

    def __init__(self, input_layer, output_layer):

        self.input_layer = input_layer
        self.output_layer = output_layer

        _validate_model(self)

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
                The str keys are the layers id, i.e. their unique identifier. The associated value
                will be a tuple of the pre and post activations.
        """

        # Input is a special case
        cached_outputs = {self.input_layer.id: self.input_layer.predict(z, output_only=output_only)}

        for layer in self.layers_by_number_of_parents[1:]:  # Input is assumed to have the least number of parents
            layer_id = layer.id

            layer_arg = self._get_layer_predict_arguments(layer, cached_outputs, output_only=output_only)

            if output_only:
                cached_outputs[layer_id] = layer.predict(layer_arg, output_only=output_only)
            else:
                pre_activation_args, post_activation_args = layer_arg
                cached_outputs[layer_id] = layer.predict(post_activation_args, output_only=output_only,
                                                         pre_activation_of_input=pre_activation_args)

        if not output_only:
            return cached_outputs
        return cached_outputs[self.output_layer.id]

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

            Notes
            -----
            This function can be written more efficiently, but at the cost of readability. I chose
            to make the function more readable
        """
        layer_args = [cached_outputs[parent.id] for parent in layer.parents]

        if output_only and len(layer.parents) == 1:
            post_activation_argument = layer_args[0]
            return post_activation_argument
        elif output_only and len(layer.parents) > 1:  # Layers assumed to have more than 1 parent
            post_activation_argument = layer_args
            return post_activation_argument
        elif not output_only and len(layer.parents) == 1:
            pre_activation_argument, post_activation_argument = layer_args[0]
            return pre_activation_argument, post_activation_argument
        else:
            pre_activation_args = [pre_activation for (pre_activation, post_activation) in layer_args]
            post_activation_args = [post_activation for (pre_activation, post_activation) in layer_args]
            return pre_activation_args, post_activation_args

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
        predictions_of_model_layers = self.predict(x_train, output_only=False)

        layer_gradients = self._get_layer_gradients(predictions_of_model_layers, y_train)
        parameter_updates = self.optimizer.step(simplify_recursive_dict(layer_gradients))
        parameter_updates_by_layer = unpack_to_recursive_dict(parameter_updates)

        for layer in self.layers_by_number_of_parents[1:]:
            if layer.id in parameter_updates_by_layer:
                layer.update_parameters_(parameter_updates_by_layer[layer.id])

    def _get_layer_gradients(self, predictions_of_model_layers, y_train):
        """ Returns the gradients for each layer as a dictionary """
        cached_pre_activation = {key: val[0] for (key, val) in predictions_of_model_layers.items()}
        cached_output = {key: val[1] for (key, val) in predictions_of_model_layers.items()}
        cached_delta = self._get_layer_deltas(y_train, cached_pre_activation, cached_output)

        grad_dict = {}
        for layer in self.layers_by_number_of_parents[1:]:
            z = tuple([cached_output[parent.id] for parent in layer.parents])
            if len(layer.parents) == 1:
                z = z[0]

            grad_dict[layer.id] = layer.get_parameter_gradients_(cached_delta[layer.id], z)

        return grad_dict

    def _get_layer_deltas(self, y_train, cached_pre_activation, cached_output):
        """ Returns the delta^k for all the layers """

        cached_delta = {}

        output_id = self.output_layer.id

        # This is for numerical stability
        if self.loss_function == 'cross_entropy':
            cached_delta[output_id] = [cached_output[output_id] - y_train]
        else:
            cached_delta[output_id] = [self._loss_function(cached_output[output_id], y_train,
                                                          grad=True)]

        # output_layer assumed to have the least amount of children
        for layer in self.layers_by_number_of_children[1:]:
            g_prime = layer.activation_function_(cached_pre_activation[layer.id], grad=True)
            z = cached_output[layer.id]

            cached_delta[layer.id] = [child.get_delta_backprop_(g_prime, cached_delta[child.id], z)
                                      for child in layer.children]

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
        full_model_dictionary = get_attributes_of_full_model(self)
        simplified_full_model_dictionary = simplify_recursive_dict(full_model_dictionary)

        with h5py.File(file_path, 'w') as h:
            for key, val in simplified_full_model_dictionary.items():
                if val is not None:
                    h.create_dataset(key, data=val)
                else:
                    h.create_dataset(key, data=np.array('null', 'S'))
