import numpy as np
import pyopencl.array as cl_array

from PySoap2_gpu import ActivationFunction


class LayerBaseAttributes:
    """ Define base attributes of Layer

        Attributes
        ----------
        input_shape : tuple
        output_shape : tuple
        activation_function : str
            Assumed to be linear activation
        built : bool
            Set to False
        device_context : pyopencl.Context
        device_queue : pyopencl.CommandQueue
    """
    def __init__(self):
        self.input_shape = ()
        self.output_shape = ()

        self.activation_function = 'linear'

        self.device_context = None
        self.device_queue = None

        self.built = False

    @property
    def activation_function_(self):
        if not ActivationFunction.initialized:
            ActivationFunction(self.device_context, self.device_queue)
        return ActivationFunction.get_activation_function(self.activation_function)

    @property
    def _memory_location(self):
        """ Return the location in memory, to be used as a unique identify for instances
            of LayerBaseAttributes

            Notes
            -----
            I'm not sure if there is anything wrong with using memory as a unique identifier,
            but another alternative, if needed, is to use an attribute bound to this class
        """
        return hex(id(self))

    @property
    def id(self):
        """ We'll use the memory location as the id of a layer """
        return f'{type(self).__name__}_{self._memory_location}'

    def get_layer_attributes_(self):
        """ Returns the attributes of the layer as a dictionary.

            Note the attributes that are returned are only the ones that can
            be saved as a hdf5 file (or converted to be saved as as hdf5)
        """
        layer_attributes = self.__dict__.copy()

        del layer_attributes['parents']
        del layer_attributes['children']
        del layer_attributes['device_context']
        del layer_attributes['device_queue']

        # Get any attributes from device back to the cpu
        for key, val in layer_attributes.items():
            if isinstance(val, cl_array.Array):
                layer_attributes[key] = val.get()

        return layer_attributes

    @property
    def input_length_device(self):
        if len(self.input_shape) == 0:
            input_length = np.array(0, dtype=np.int32)
        else:
            input_length = np.array(np.prod(self.input_shape), dtype=np.int32)

        return cl_array.to_device(self.device_queue, input_length)

    @property
    def output_length_device(self):
        if len(self.output_shape) == 0:
            output_length = np.array(0, dtype=np.int32)
        else:
            output_length = np.array(np.prod(self.output_shape), dtype=np.int32)

        return cl_array.to_device(self.device_queue, output_length)
