import numpy as np

import pyopencl as cl
import pyopencl.array as cl_array

from PySoap2.models import Model as CpuBaseModel

from PySoap2_gpu.optimizers import Optimizer, get_optimizer
from PySoap2_gpu.utils.dictionary_tricks import simplify_recursive_dict, unpack_to_recursive_dict

from PySoap2_gpu.functions import ErrorFunction, MetricFunction


class Model(CpuBaseModel):
    def __init__(self, input_layer, output_layer, device_context=None, device_queue=None):
        CpuBaseModel.__init__(self, input_layer, output_layer)

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
        if isinstance(optimizer, Optimizer):
            self.optimizer = optimizer
        elif isinstance(optimizer, str):
            self.optimizer = get_optimizer(optimizer)
        else:
            raise ValueError("optimizer must be an instance of Optimizer or str")

        self.loss_function = loss_function
        self.metric_function = metrics

        for layer in self.layers_by_number_of_parents:
            layer.build(self.device_context, self.device_queue)

    def predict(self, z, output_only=True):
        """ Forward propagates the input

            Parameters
            ----------
            z : np.array or cl_array.Array
            output_only : bool

            Notes
            -----
            If z is a np.array then it will be converted to be a cl_array.Array
        """

        if not isinstance(z, cl_array.Array):
            z = z.astype(np.float32)
            z = cl_array.to_device(self.device_queue, z)

        return super().predict(z, output_only=output_only)

    def evaluate(self, x_test, y_test):
        """ Evaluates the model with the given loss-function and metric function

            Parameters
            ----------
            x_test : np.array
            y_test : np.array
        """

        x_test = x_test.astype(np.float32)
        y_test = y_test.astype(np.float32)

        x_test_device = cl_array.to_device(self.device_queue, x_test)
        y_test_device = cl_array.to_device(self.device_queue, y_test)

        prediction = self.predict(x_test_device)

        loss_val = self._loss_function(prediction, y_test_device)

        if isinstance(loss_val, cl_array.Array):
            loss_val = loss_val.get()

        eval_str = f'{self.loss_function}: {format(loss_val, ".4f")}'

        if self.metric_function is not None:
            metric_val = self._metric(prediction, y_test_device)
            if isinstance(metric_val, cl_array.Array):
                metric_val = metric_val.get()

            eval_str += f' - {self.metric_function}: {format(metric_val, ".4f")}'

        return eval_str

    def _back_prop(self, x_train, y_train):
        """ Perform one iteration of backpropagation on the given batches

            Parameters
            ----------
            x_train : np.array
            y_train : np.array
        """

        x_train = x_train.astype(np.float32)
        y_train = y_train.astype(np.float32)

        y_train_device = cl_array.to_device(self.device_queue, y_train)

        predictions_of_model_layers = self.predict(x_train, output_only=False)

        layer_gradients = self._get_layer_gradients(predictions_of_model_layers, y_train_device)
        parameter_updates = self.optimizer.step(simplify_recursive_dict(layer_gradients))
        parameter_updates_by_layer = unpack_to_recursive_dict(parameter_updates)

        for layer in self.layers_by_number_of_parents[1:]:
            if layer.id in parameter_updates_by_layer:
                layer.update_parameters_(parameter_updates_by_layer[layer.id])

    @property
    def _loss_function(self):
        return ErrorFunction.get_error_function(self.loss_function)

    @property
    def _metric(self):
        if self.metric_function is not None:
            return MetricFunction.get_metric_function(self.metric_function)
        return None
