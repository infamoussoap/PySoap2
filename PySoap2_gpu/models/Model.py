import pyopencl as cl
import pyopencl.array as cl_array

from PySoap2.models import Model as CpuBaseModel

from PySoap2_gpu.optimizers import Optimizer, get_optimizer
from PySoap2_gpu.utils.dictionary_tricks import simplify_recursive_dict, unpack_to_recursive_dict

from PySoap2_gpu.functions import ErrorFunction, MetricFunction


class Model(CpuBaseModel):
    def __init__(self, input_layer, output_layer):
        CpuBaseModel.__init__(self, input_layer, output_layer)

        platform = cl.get_platforms()[0]
        device = platform.get_devices()[0]

        self.device_context = cl.Context([device])
        self.device_queue = cl.CommandQueue(self.device_context)

        if ErrorFunction.initialized:
            ErrorFunction(self.device_context, self.device_queue)

        if MetricFunction.initialized:
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

    def predict_and_send_input_to_device(self, z_cpu, output_only=True):
        z_device = cl_array.to_device(self.device_queue, z_cpu)
        return super().predict(z_device, output_only=output_only)

    def predict_and_covert_output_to_cpu(self, z_cpu, output_only=True):
        cached_output_on_device = self.predict_and_send_input_to_device(z_cpu, output_only=output_only)

        if output_only:
            return cached_output_on_device.get()
        else:
            return {key: val.get() for key, val in cached_output_on_device.items()}

    def evaluate(self, x_test, y_test):
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
        """ Perform one iteration of backpropagation on the given batches """
        y_train_device = cl_array.to_device(self.device_queue, y_train)

        predictions_of_model_layers = self.predict_and_send_input_to_device(x_train, output_only=False)

        layer_gradients = self._get_layer_gradients(predictions_of_model_layers, y_train_device)
        parameter_updates = self.optimizer.step(simplify_recursive_dict(layer_gradients))
        parameter_updates_by_layer = unpack_to_recursive_dict(parameter_updates)

        for layer in self.layers_by_number_of_parents[1:]:
            if layer.memory_location in parameter_updates_by_layer:
                layer.update_parameters_(parameter_updates_by_layer[layer.memory_location])

    @property
    def _loss_function(self):
        return ErrorFunction.get_error_function(self.loss_function)

    @property
    def _metric(self):
        if self.metric_function is not None:
            return MetricFunction.get_metric_function(self.metric_function)
        return None
