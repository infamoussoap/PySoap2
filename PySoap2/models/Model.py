import numpy as np
import functools
import random

import h5py

from PySoap2.optimizers import Optimizer, get_optimizer
from PySoap2 import get_error_function, get_metric_function
from PySoap2.layers import Layer

from .Logger import ModelLogger
from .dictionary_tricks import simplify_recursive_dict, unpack_to_recursive_dict
from .SaveModel import get_attributes_of_full_model


def _validate_model(model):
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


class Model:
    def __init__(self, input_layer, output_layers):

        self.input_layer = input_layer
        self.output_layers = as_list_of_data_type(output_layers, Layer, 'output_layers')
        self.output_length = len(self.output_layers)

        _validate_model(self)

        self.optimizer = None
        self.loss_functions = None
        self.metric_functions = None

    def build(self, loss_function, optimizer, metrics=None):
        """ Build the layers in the tree network, and save attributes of Model

            Parameters
            ----------
            loss_function : str or list[str]
            optimizer : str or :obj:Optimizer
            metrics : None or str or list[str], optional
        """
        self._set_optimizer(optimizer)
        self._set_loss_functions(loss_function)
        self._set_metric_functions(metrics)

        for layer in self.layers_by_number_of_parents:
            layer.build()

    def _set_optimizer(self, optimizer):
        if isinstance(optimizer, Optimizer):
            self.optimizer = optimizer
            return
        elif isinstance(optimizer, str):
            self.optimizer = get_optimizer(optimizer)
            return
        raise ValueError("optimizer must be an instance of Optimizer or str")

    def _set_loss_functions(self, loss_function):
        if isinstance(loss_function, str):
            self.loss_functions = [loss_function] * self.output_length
        else:
            self.loss_functions = as_list_of_data_type(loss_function, str, 'loss_function')

    def _set_metric_functions(self, metrics):
        if metrics is None:
            self.metric_functions = [None] * self.output_length
        elif isinstance(metrics, str):
            self.metric_functions = [metrics] * self.output_length
        elif all([metric is None or isinstance(metric, str) for metric in metrics]):
            self.metric_functions = metrics
        else:
            raise ValueError('metrics need to be None, or string, or a list of None/str.')

    @property
    @functools.lru_cache()
    def layers_by_number_of_parents(self):
        """ Returns a list by order of the nodes with the least amount of parents to the most parents """
        current_layers = list(self.output_layers)  # Terminal node will have the most parents
        layer_order = []

        while len(current_layers) > 0:
            for layer in current_layers:
                if layer in layer_order:
                    layer_order.remove(layer)
                layer_order.append(layer)

            parents_of_current_layers = [layer.parents for layer in current_layers]
            current_layers = set(functools.reduce(lambda x, y: x + y, parents_of_current_layers))

        return layer_order[::-1]

    def predict(self, z):
        """ Perform forward propagation of the whole network

            Parameters
            ----------
            z : (N, *input_shape) np.array
                The Input

            Returns
            -------
            list[np.ndarray] or np.ndarray
                Returns a list of np.ndarray if there is multiple outputs, otherwise it will return the single
                np.ndarray
        """

        cached_layer_outputs = self._get_outputs_of_layers(z, output_only=True, training=False)

        if self.output_length == 1:
            return cached_layer_outputs[self.output_layers[0]]
        return [cached_layer_outputs[layer] for layer in self.output_layers]

    def _predict_as_list(self, z):
        """ Perform forward propagation of the whole network

            Parameters
            ----------
            z : (N, *input_shape) np.array
                The Input

            Returns
            -------
            list[np.ndarray]
                Will always return a list of np.ndarray even if there is only 1 output

        """
        cached_outputs = self._get_outputs_of_layers(z, output_only=True, training=False)
        return [cached_outputs[output_layer.id] for output_layer in self.output_layers]

    def _get_outputs_of_layers(self, z, output_only=True, training=False):
        """ Perform forward propagation of the whole network

            Parameters
            ----------
            z : (N, *input_shape) np.array
                The Input
            output_only : bool
                If true then only the outputs (i.e. post-activations) of the layers will be returned
                Otherwise the pre and post activations will be returned
            training : bool
                Some layers (like the dropout layer) have different behaviour when it is training

            Returns
            -------
            dict of str - np.ndarray
                This is returned if output_only=True
                The str keys are the layers id, i.e. their unique identifier. The associated value
                will be the post-activations.

            dict of str - (np.ndarray, np.ndarray)
                This is returned if output_only=False
                The str keys are the layers id, i.e. their unique identifier. The associated value
                will be a tuple of the pre and post activations.
        """

        # Input is a special case
        cached_outputs = {self.input_layer.id: self.input_layer.predict(z, output_only=output_only)}

        for layer in self.layers_by_number_of_parents[1:]:  # Input is assumed to have the least number of parents
            layer_id = layer.id
            layer_arg = self._get_layer_predict_arguments(layer, cached_outputs, output_only=output_only)

            if output_only:
                cached_outputs[layer_id] = layer.predict(layer_arg, output_only=output_only, training=training)
            else:
                pre_activation_args, post_activation_args = layer_arg
                cached_outputs[layer_id] = layer.predict(post_activation_args,
                                                         pre_activation_of_input=pre_activation_args,
                                                         output_only=output_only, training=training)
        return cached_outputs

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
            y_test : np.array or list[np.array] or tuple[np.array]
                y_test is the associated list of outputs to the list of inputs X_test.

            Returns
            -------
            str
                The error
        """
        y_test_as_list = as_list_of_data_type(y_test, np.ndarray, 'y_test')
        check_valid_targets_length(y_test_as_list, self.output_length)

        predictions = self._predict_as_list(x_test)
        loss_vals = self._loss_function_as_list(predictions, y_test_as_list)

        eval_str = f'total loss : {format(np.sum(loss_vals), ".4f")}'
        for (loss_function_name, val) in zip(self.loss_functions, loss_vals):
            eval_str += f' - {loss_function_name}_loss : {format(val, ".4f")}'

        metric_vals = self._metric_as_list(predictions, y_test_as_list)
        for (metric_function_name, val) in zip(self.metric_functions, metric_vals):
            if metric_function_name is not None:
                eval_str += f' - {metric_function_name} : {format(val, ".4f")}'

        return eval_str

    def train(self, x_train, y_train, epochs=100, batch_size=None, verbose=True, log=False,
              x_test=None, y_test=None):
        """ Train the neural network by means of back propagation
            Parameters
            ----------
            x_train : np.array
                x_train is assumed to be a list of all the inputs to be forward propagated. In particular
                it is assumed that the first index of x_train is the index that inputs is accessed by
            y_train : np.array or list[np.array] or tuple[np.array]
                y_train is the associated list of outputs to the list of inputs x_train. More specifically,
                the neural network will be trained to find the map x_train -> y_train

                y_train should be a list or tuple if the network has multiple outputs
            epochs : :obj:`int`, optional
                Number of times the neural network will see the entire dataset
            batch_size : :obj:`int`, optional
                The batch size for gradient descent. If not defined then `batch_size` is set to the
                length of the dataset, i.e. traditional gradient descent.
            verbose : bool, optional
                If set to `True` then the model performance will be printed after each epoch
            log : bool, optional
                Log the training history
            x_test : np.array, optional
            y_test : np.array, optional
        """
        y_train_as_list = as_list_of_data_type(y_train, np.ndarray, 'y_train')
        check_valid_targets_length(y_train_as_list, self.output_length)
        model_logger = log if isinstance(log, ModelLogger) else ModelLogger(self, x_train, y_train_as_list,
                                                                            x_test=x_test, y_test=y_test)

        training_length = len(x_train)
        if batch_size is None:
            batch_size = training_length
        index = list(range(training_length))

        for epoch in range(epochs):
            if verbose:
                print(f'Training on {len(x_train)} samples')

            random.shuffle(index)
            for i in range(np.ceil(training_length / batch_size).astype(int)):
                start, end = i * batch_size, (i + 1) * batch_size
                batch_x, batch_y = x_train[index[start:end]], [y[index[start:end]] for y in y_train_as_list]

                self._back_prop(batch_x, batch_y)

            if verbose:
                start, end = 0, batch_size
                batch_x, batch_y = x_train[index[start:end]], [y[index[start:end]] for y in y_train_as_list]
                evaluation = self.evaluate(batch_x, batch_y)
                print(f'Epoch {epoch + 1}/{epochs}')
                print(evaluation)

            if log:
                model_logger.log_model(epoch + 1, None)

        if log and model_logger.auto_save:
            model_logger.save()

    def _back_prop(self, x_train, y_train_as_list):
        """ Perform one iteration of backpropagation on the given batches

            Parameters
            ----------
            x_train : np.ndarray
            y_train_as_list : list[np.ndarray]
                It is assumed that y_train is a list of np.ndarray, and should be pre-processed before it is
                passed into this method
        """
        predictions_of_model_layers = self._get_outputs_of_layers(x_train, output_only=False, training=True)

        layer_gradients = self._get_layer_gradients(predictions_of_model_layers, y_train_as_list)
        parameter_updates = self.optimizer.step(simplify_recursive_dict(layer_gradients))
        parameter_updates_by_layer = unpack_to_recursive_dict(parameter_updates)

        for layer in self.layers_by_number_of_parents[1:]:
            if layer.id in parameter_updates_by_layer:
                layer.update_parameters_(parameter_updates_by_layer[layer.id])

    def _get_layer_gradients(self, predictions_of_model_layers, y_train_as_list):
        """ Returns the gradients for each layer as a dictionary """
        cached_pre_activation = {key: val[0] for (key, val) in predictions_of_model_layers.items()}
        cached_output = {key: val[1] for (key, val) in predictions_of_model_layers.items()}
        cached_delta = self._get_layer_deltas(y_train_as_list, cached_pre_activation, cached_output)

        grad_dict = {}
        for layer in self.layers_by_number_of_parents[1:]:
            z = tuple([cached_output[parent.id] for parent in layer.parents])
            if len(layer.parents) == 1:
                z = z[0]

            grad_dict[layer.id] = layer.get_parameter_gradients_(cached_delta[layer.id], z)

        return grad_dict

    def _get_layer_deltas(self, y_train_as_list, cached_pre_activation, cached_output):
        """ Returns the delta^k for all the layers """

        cached_delta = {}

        delta_for_output_layers = self._get_delta_for_output_layers(y_train_as_list, cached_pre_activation,
                                                                    cached_output)
        cached_delta.update(delta_for_output_layers)

        # output layers assumed to have the least amount of children and can be skipped
        for layer in self.layers_by_number_of_children[self.output_length:]:
            g_prime = layer.activation_function_(cached_pre_activation[layer.id], grad=True)
            z = cached_output[layer.id]

            cached_delta[layer.id] = [child.get_delta_backprop_(g_prime, cached_delta[child.id], z)
                                      for child in layer.children]

        return cached_delta

    def _get_delta_for_output_layers(self, y_train_as_list, cached_pre_activation, cached_output):
        cached_delta = {}
        for (output_layer, loss_name, y_train) in zip(self.output_layers, self.loss_functions, y_train_as_list):
            output_id = output_layer.id

            # This is for numerical stability
            if loss_name == "cross_entropy":
                cached_delta[output_id] = [cached_output[output_id] - y_train]
            else:
                loss_function = get_error_function(loss_name)

                ds_dz = loss_function(cached_output[output_id], y_train, grad=True)
                g_prime = output_layer.activation_function_(cached_pre_activation[output_id], grad=True)
                cached_delta[output_id] = [ds_dz * g_prime]
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

    def _loss_function(self, predictions, targets, grad=False):
        loss = self._loss_function_as_list(predictions, targets, grad=grad)
        if self.output_length == 1:
            return loss[0]
        return loss

    def _loss_function_as_list(self, predictions, targets, grad=False):
        return [get_error_function(name)(prediction, target, grad=grad)
                for (name, prediction, target) in zip(self.loss_functions, predictions, targets)]

    def _metric(self, predictions, targets):
        metric_vals = self._metric_as_list(predictions, targets)
        if self.output_length == 1:
            return metric_vals[0]
        return metric_vals

    def _metric_as_list(self, predictions, targets):
        return [None if name is None else get_metric_function(name)(prediction, target)
                for (name, prediction, target) in zip(self.metric_functions, predictions, targets)]

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


def as_list_of_data_type(val, data_type, data_name):
    if isinstance(val, data_type):
        return [val]
    elif all([isinstance(v, data_type) for v in val]):
        return val
    raise ValueError(f'{data_name} needs to be an instance of {data_type.__name__}, or a list of'
                     f' {data_type.__name__}')


def check_valid_targets_length(targets_as_list, output_length):
    if len(targets_as_list) != output_length:
        raise ValueError(f'Expecting {output_length} targets, but got {len(targets_as_list)}')
