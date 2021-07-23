import numpy as np

import pyopencl as cl
import pyopencl.array as cl_array

from PySoap2.models import Model as CpuBaseModel
from PySoap2.models import ModelLogger
from PySoap2.utils import ImageAugmentationGenerator

from PySoap2_gpu.layers import Layer

from PySoap2_gpu.optimizers import GPUOptimizer, get_optimizer
from PySoap2_gpu.utils.dictionary_tricks import simplify_recursive_dict, unpack_to_recursive_dict

from PySoap2_gpu.functions import ErrorFunction, MetricFunction

from .ValueChecks import as_list_of_data_types
from .ValueChecks import as_list_of_clarrays
from .ValueChecks import convert_to_clarray
from PySoap2.models.ValueChecks import check_valid_targets_length
from PySoap2.models.ValueChecks import validate_model


class Model(CpuBaseModel):
    def __init__(self, input_layer, output_layers, device_context=None, device_queue=None):
        self.input_layer = input_layer
        self.output_layers = as_list_of_data_types(output_layers, Layer, 'output_layers')
        self.output_length = len(self.output_layers)

        validate_model(self)

        self.optimizer = None
        self.loss_functions = None
        self.metric_functions = None

        if device_queue is None and device_context is None:
            platform = cl.get_platforms()[0]
            device = platform.get_devices()[0]

            self.device_context = cl.Context([device])
            self.device_queue = cl.CommandQueue(self.device_context)
        else:
            self.device_queue = device_queue
            self.device_context = device_context

        if not ErrorFunction.initialized:
            ErrorFunction(self.device_context, self.device_queue)

        if not MetricFunction.initialized:
            MetricFunction(self.device_context, self.device_queue)

    def build(self, loss_function, optimizer, metrics=None):
        self._set_optimizer(optimizer)
        self._set_loss_functions(loss_function)
        self._set_metric_functions(metrics)

        for layer in self.layers_by_number_of_parents:
            layer.build(self.device_context, self.device_queue)

    def _set_optimizer(self, optimizer):
        if isinstance(optimizer, GPUOptimizer):
            self.optimizer = optimizer
        elif isinstance(optimizer, str):
            self.optimizer = get_optimizer(optimizer)
        else:
            raise ValueError("Optimizer must be instance of GPUOptimizer or str")

    def predict(self, z, output_only=True, training=False):
        """ Forward propagates the input

            Parameters
            ----------
            z : np.array or cl_array.Array
            output_only : bool
            training : bool
                Some layers behave differently during the training phase

            Notes
            -----
            If z is a np.array then it will be converted to be a cl_array.Array
        """
        z = convert_to_clarray(self.device_queue, z, dtype=np.float64)
        return super().predict(z)

    def evaluate(self, x_test, y_test):
        """ Evaluates the model with the given loss-function and metric function

            Parameters
            ----------
            x_test : np.array or cl_array.Array
            y_test : np.array or cl_array.Array
        """

        x_test = convert_to_clarray(self.device_queue, x_test, dtype=np.float64)
        y_test_as_list = as_list_of_clarrays(self.device_queue, y_test, 'y_test', dtype=np.float64)
        check_valid_targets_length(y_test_as_list, self.output_length, 'y_test')

        return self._evaluate(x_test, y_test_as_list)

    def _evaluate(self, x_test, y_test_as_list):
        """ Evaluates the model

            Parameters
            ----------
            x_test : cl_array.Array
            y_test_as_list : list[cl_array.Array]

            Returns
            -------
            str
        """
        predictions = self._predict_as_list(x_test)
        loss_vals = self._loss_function_as_list(predictions, y_test_as_list, grad=False)
        total_loss = sum(val.get() for val in loss_vals)

        eval_str = f"total loss : {format(total_loss, '.4f')}"
        for (loss_function_name, val) in zip(self.loss_functions, loss_vals):
            eval_str += f" - {loss_function_name}_loss : {format(val.get(), '.4f')}"

        metric_vals = self._metric_as_list(predictions, y_test_as_list)
        for (metric_function_name, val) in zip(self.metric_functions, metric_vals):
            if metric_function_name is not None:
                eval_str += f" - {metric_function_name} : {format(val, '.4f')}"

        return eval_str

    def train(self, x_train, y_train, epochs=100, batch_size=None, verbose=True, log=False,
              x_test=None, y_test=None):
        """ (x_train, y_train) need to be np.array for ImageAugmentation to for

            (x_test, y_test) assumed to be np.array or cl_array.Array
        """
        if x_test is not None:
            x_test = convert_to_clarray(self.device_queue, x_test, dtype=np.float64)
        if y_test is not None:
            y_test = as_list_of_clarrays(self.device_queue, y_test, 'y_test', dtype=np.float64)
            check_valid_targets_length(y_test, self.output_length, 'y_test')

        # If you want to log, but didn't pass in a logger
        if log and not isinstance(log, ModelLogger):
            x_train_device = convert_to_clarray(self.device_queue, x_train, dtype=np.float64)
            y_train_device = as_list_of_clarrays(self.device_queue, y_train, 'y_train', dtype=np.float64)
            check_valid_targets_length(y_train_device, self.output_length, 'y_train')

            log = ModelLogger(self, x_train_device, y_train_device, x_test=x_test, y_test=y_test)

        if not isinstance(x_train, (np.ndarray, ImageAugmentationGenerator)):
            raise ValueError("x_train needs to be an instance of np.ndarray or ImageAugmentationGenerator")
        y_train_as_list = as_list_of_data_types(y_train, np.ndarray, 'y_train')
        super()._train(x_train, y_train_as_list, epochs, batch_size, verbose, log)

    def _back_prop(self, x_train, y_train):
        """ Perform one iteration of backpropagation on the given batches

            Parameters
            ----------
            x_train : np.array or cl_array.Array
            y_train : np.array or cl_array.Array
        """
        x_train = convert_to_clarray(self.device_queue, x_train, dtype=np.float64)
        y_train = as_list_of_clarrays(self.device_queue, y_train, 'y_train', dtype=np.float64)

        predictions_of_model_layers = self._get_outputs_of_layers(x_train, output_only=False, training=True)

        layer_gradients = self._get_layer_gradients(predictions_of_model_layers, y_train)
        parameter_updates = self.optimizer.step(simplify_recursive_dict(layer_gradients))
        parameter_updates_by_layer = unpack_to_recursive_dict(parameter_updates)

        for layer in self.layers_by_number_of_parents[1:]:
            if layer.id in parameter_updates_by_layer:
                layer.update_parameters_(parameter_updates_by_layer[layer.id])

    def _loss_function(self, predictions, targets, grad=False):
        loss = self._loss_function_as_list(predictions, targets, grad=grad)
        if self.output_length == 1:
            return loss[0]
        return loss

    def _loss_function_as_list(self, predictions, targets, grad=False):
        predictions = as_list_of_clarrays(self.device_queue, predictions, 'predictions', dtype=np.float64)
        targets = as_list_of_clarrays(self.device_queue, targets, 'targets', dtype=np.float64)

        return [ErrorFunction.get_error_function(name)(prediction, target, grad=grad)
                for (name, prediction, target) in zip(self.loss_functions, predictions, targets)]

    def _metric(self, predictions, targets):
        metric_vals = self._metric_as_list(predictions, targets)
        if self.metric_functions is None:
            return None

        if self.output_length == 1:
            return metric_vals[0]
        return metric_vals

    def _metric_as_list(self, predictions, targets):
        predictions = as_list_of_clarrays(self.device_queue, predictions, 'predictions', dtype=np.float64)
        targets = as_list_of_clarrays(self.device_queue, targets, 'targets', dtype=np.float64)

        return [None if name is None else MetricFunction.get_metric_function(name)(prediction, target)
                for (name, prediction, target) in zip(self.metric_functions, predictions, targets)]
