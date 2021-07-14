import numpy as np

import pyopencl as cl
import pyopencl.array as cl_array

from PySoap2.models import Model as CpuBaseModel
from PySoap2.models import ModelLogger
from PySoap2.utils import ImageAugmentationGenerator

from PySoap2_gpu.optimizers import GPUOptimizer, get_optimizer
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
        if isinstance(optimizer, GPUOptimizer):
            self.optimizer = optimizer
        elif isinstance(optimizer, str):
            self.optimizer = get_optimizer(optimizer)
        else:
            raise ValueError("optimizer must be an instance of GPUOptimizer or str")

        self.loss_function = loss_function
        self.metric_function = metrics

        for layer in self.layers_by_number_of_parents:
            layer.build(self.device_context, self.device_queue)

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
        z = convert_to_clarray(self.device_queue, z, dtype=np.float32)

        return super().predict(z, output_only=output_only, training=training)

    def evaluate(self, x_test, y_test):
        """ Evaluates the model with the given loss-function and metric function

            Parameters
            ----------
            x_test : np.array or cl_array.Array
            y_test : np.array or cl_array.Array
        """
        x_test = convert_to_clarray(self.device_queue, x_test, dtype=np.float32)
        y_test = convert_to_clarray(self.device_queue, y_test, dtype=np.float32)

        prediction = self.predict(x_test)

        loss_val = self._loss_function(prediction, y_test)

        if isinstance(loss_val, cl_array.Array):
            loss_val = loss_val.get()

        eval_str = f'{self.loss_function}: {format(loss_val, ".4f")}'

        if self.metric_function is not None:
            metric_val = self._metric(prediction, y_test)
            if isinstance(metric_val, cl_array.Array):
                metric_val = metric_val.get()

            eval_str += f' - {self.metric_function}: {format(metric_val, ".4f")}'

        return eval_str

    def train(self, x_train, y_train, epochs=100, batch_size=None, verbose=True, log=False,
              x_test=None, y_test=None):
        """ (x_train, y_train) need to be np.array for ImageAugmentation to for

            (x_test, y_test) assumed to be np.array or cl_array.Array
        """
        if x_test is not None:
            x_test = convert_to_clarray(self.device_queue, x_test, dtype=np.float32)
        if y_test is not None:
            y_test = convert_to_clarray(self.device_queue, y_test, dtype=np.float32)

        # If you want to log, but didn't pass in a logger
        if log and not isinstance(log, ModelLogger):
            x_train_device = convert_to_clarray(self.device_queue, x_train, dtype=np.float32)
            y_train_device = convert_to_clarray(self.device_queue, y_train, dtype=np.float32)
            log = ModelLogger(self, x_train_device, y_train_device, x_test=x_test, y_test=y_test)

        super().train(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose,
                      log=log, x_test=x_test, y_test=y_test)

    def _back_prop(self, x_train, y_train):
        """ Perform one iteration of backpropagation on the given batches

            Parameters
            ----------
            x_train : np.array or cl_array.Array
            y_train : np.array or cl_array.Array
        """
        x_train = convert_to_clarray(self.device_queue, x_train, dtype=np.float32)
        y_train = convert_to_clarray(self.device_queue, y_train, dtype=np.float32)

        predictions_of_model_layers = self.predict(x_train, output_only=False, training=True)

        layer_gradients = self._get_layer_gradients(predictions_of_model_layers, y_train)
        parameter_updates = self.optimizer.step(simplify_recursive_dict(layer_gradients))
        parameter_updates_by_layer = unpack_to_recursive_dict(parameter_updates)

        for layer in self.layers_by_number_of_parents[1:]:
            if layer.id in parameter_updates_by_layer:
                layer.update_parameters_(parameter_updates_by_layer[layer.id])

    @property
    def _loss_function(self):
        return self._wrapped_loss_function

    def _wrapped_loss_function(self, *args):
        """ args to the loss_function needs to be cl_arrays """
        args = [convert_to_clarray(self.device_queue, arg) for arg in args]
        return ErrorFunction.get_error_function(self.loss_function)(*args)

    @property
    def _metric(self):
        if self.metric_function is not None:
            return self._wrapped_metric
        return None

    def _wrapped_metric(self, *args):
        args = [convert_to_clarray(self.device_queue, arg) for arg in args]
        return MetricFunction.get_metric_function(self.metric_function)(*args)


def convert_to_clarray(device_queue, array, dtype=np.float32):
    """ Converts the array to cl_array.Array

        Parameters
        ----------
        device_queue : cl.CommandQueue
            The queue for the device to put the array on
        array : np.array or cl_array.Array
            The array to be converted
        dtype : Class, optional
            The data type for the array

        Notes
        -----
        If array is a cl_array.Array then it will be returned
        If array is np.array then it will be converted to the dtype and sent to the queue
    """
    if isinstance(array, cl_array.Array):
        return array.astype(dtype)
    elif isinstance(array, np.ndarray):
        array = convert_to_contiguous_array(array)
        return cl_array.to_device(device_queue, array.astype(dtype))
    elif isinstance(array, ImageAugmentationGenerator):
        images = convert_to_contiguous_array(array.images)
        return cl_array.to_device(device_queue, images.astype(dtype))
    else:
        raise ValueError(f'{type(array)} not supported, only cl_array and np.ndarray allowed.')


def convert_to_contiguous_array(array):
    if not is_array_contiguous(array):
        return np.ascontiguousarray(array)
    return array


def is_array_contiguous(array):
    """ array assumed to be np.array """
    return array.flags.forc
