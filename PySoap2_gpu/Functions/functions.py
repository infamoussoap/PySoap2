from PySoap2_gpu.utils import ClMathFunctions
from PySoap2_gpu.utils import ClArrayTricks

from .ActivationFunctions import activation_functions
from .ErrorFunctions import error_functions
from .MetricFunctions import metric_functions


class ActivationFunction:
    initialized = False

    device_context = None
    device_queue = None

    def __init__(self, device_context, device_queue):
        # If this class is initialized, it means that the programs is already on the device
        if ActivationFunction.initialized:
            return

        ActivationFunction.device_context = device_context
        ActivationFunction.device_queue = device_queue

        ActivationFunction.initialized = True

        ClMathFunctions(device_context, device_queue)
        ClArrayTricks(device_context, device_queue)

    @staticmethod
    def get_activation_function(name):
        function = activation_functions.get(name, None)
        if function is None:
            raise Exception(f'{name} is not a defined function.')
        return function


class ErrorFunction:
    initialized = False

    device_context = None
    device_queue = None

    def __init__(self, device_context, device_queue):
        if ErrorFunction.initialized:
            return

        ErrorFunction.device_queue = device_queue
        ErrorFunction.device_context = device_context

        ClMathFunctions(device_context, device_queue)
        ClArrayTricks(device_context, device_queue)

        ErrorFunction.initialized = True

    @staticmethod
    def get_error_function(name):
        function = error_functions.get(name, None)
        if function is None:
            raise Exception(f'{name} is not a defined error function.')
        return function


class MetricFunction:
    initialized = False

    device_context = None
    device_queue = None

    def __init__(self, device_context, device_queue):
        if MetricFunction.initialized:
            return

        MetricFunction.device_queue = device_queue
        MetricFunction.device_context = device_context

        ClArrayTricks(device_context, device_queue)

        MetricFunction.initialized = True

    @staticmethod
    def get_metric_function(name):
        function = metric_functions.get(name, None)
        if function is None:
            raise Exception(f'{name} is not a defined metric.')
        return function
