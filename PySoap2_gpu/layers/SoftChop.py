import numpy as np
from functools import reduce

import pyopencl as cl
import pyopencl.array as cl_array

from .c_code.multi_softchop_c_code import multi_softchop_source_code
from .c_code.softchop_c_code import softchop_source_code

from PySoap2_gpu.layers import Layer
from PySoap2_gpu.layers import NetworkNode
from PySoap2_gpu.layers.LayerBaseAttributes import LayerBaseAttributes

from .ValueChecks import assert_instance_of_cl_array
from .ValueChecks import check_built

from PySoap2_gpu.utils import ClArrayTricks


class MultiSoftChop:
    """ The interface between the compiled pyopencl-c code with python

        Notes
        -----
        Arguments to all methods are assumed to be stored on the device
    """

    device_context = None
    device_queue = None

    device_program = None

    initialized = False

    def __init__(self, device_context, device_queue):
        if MultiSoftChop.initialized:
            return

        MultiSoftChop.device_context = device_context
        MultiSoftChop.device_queue = device_queue

        MultiSoftChop.device_program = cl.Program(device_context, multi_softchop_source_code).build()

        MultiSoftChop.initialized = True

    @staticmethod
    def eval(x_device, a1_device, a2_device, epsilon1_device, epsilon2_device):
        out_device = cl_array.empty_like(x_device)

        input_length = np.array(np.prod(a1_device.shape)).astype(np.int32)
        N = np.array(len(x_device)).astype(np.int32)

        input_length_device = cl_array.to_device(MultiSoftChop.device_queue, input_length)

        args = [x_device, a1_device, a2_device, epsilon1_device, epsilon2_device, input_length_device, out_device]
        args_data = [arg.data for arg in args]

        event = MultiSoftChop.device_program.softchop_eval(MultiSoftChop.device_queue, (N, input_length), None,
                                                           *args_data)
        event.wait()

        return out_device

    @staticmethod
    def dx(x_device, a1_device, a2_device, epsilon1_device, epsilon2_device):
        out_device = cl_array.empty_like(x_device)

        input_length = np.array(np.prod(a1_device.shape)).astype(np.int32)
        N = np.array(len(x_device)).astype(np.int32)

        input_length_device = cl_array.to_device(MultiSoftChop.device_queue, input_length)

        args = [x_device, a1_device, a2_device, epsilon1_device, epsilon2_device, input_length_device, out_device]
        args_data = [arg.data for arg in args]

        event = MultiSoftChop.device_program.softchop_dx(MultiSoftChop.device_queue, (N, input_length), None,
                                                         *args_data)
        event.wait()

        return out_device

    @staticmethod
    def da1(x_device, a1_device, a2_device, epsilon1_device, epsilon2_device):
        out_device = cl_array.empty_like(x_device)

        input_length = np.array(np.prod(a1_device.shape)).astype(np.int32)
        N = np.array(len(x_device)).astype(np.int32)

        input_length_device = cl_array.to_device(MultiSoftChop.device_queue, input_length)

        args = [x_device, a1_device, a2_device, epsilon1_device, epsilon2_device, input_length_device, out_device]
        args_data = [arg.data for arg in args]

        event = MultiSoftChop.device_program.softchop_da1(MultiSoftChop.device_queue, (N, input_length), None,
                                                          *args_data)
        event.wait()

        return out_device

    @staticmethod
    def da2(x_device, a1_device, a2_device, epsilon1_device, epsilon2_device):
        out_device = cl_array.empty_like(x_device)

        input_length = np.array(np.prod(a1_device.shape)).astype(np.int32)
        N = np.array(len(x_device)).astype(np.int32)

        input_length_device = cl_array.to_device(MultiSoftChop.device_queue, input_length)

        args = [x_device, a1_device, a2_device, epsilon1_device, epsilon2_device, input_length_device, out_device]
        args_data = [arg.data for arg in args]

        event = MultiSoftChop.device_program.softchop_da2(MultiSoftChop.device_queue, (N, input_length), None,
                                                          *args_data)
        event.wait()

        return out_device

    @staticmethod
    def de1(x_device, a1_device, a2_device, epsilon1_device, epsilon2_device):
        out_device = cl_array.empty_like(x_device)

        input_length = np.array(np.prod(a1_device.shape)).astype(np.int32)
        N = np.array(len(x_device)).astype(np.int32)

        input_length_device = cl_array.to_device(MultiSoftChop.device_queue, input_length)

        args = [x_device, a1_device, a2_device, epsilon1_device, epsilon2_device, input_length_device, out_device]
        args_data = [arg.data for arg in args]

        event = MultiSoftChop.device_program.softchop_de1(MultiSoftChop.device_queue, (N, input_length), None,
                                                          *args_data)
        event.wait()

        return out_device

    @staticmethod
    def de2(x_device, a1_device, a2_device, epsilon1_device, epsilon2_device):
        out_device = cl_array.empty_like(x_device)

        input_length = np.array(np.prod(a1_device.shape)).astype(np.int32)
        N = np.array(len(x_device)).astype(np.int32)

        input_length_device = cl_array.to_device(MultiSoftChop.device_queue, input_length)

        args = [x_device, a1_device, a2_device, epsilon1_device, epsilon2_device, input_length_device, out_device]
        args_data = [arg.data for arg in args]

        event = MultiSoftChop.device_program.softchop_de2(MultiSoftChop.device_queue, (N, input_length), None,
                                                          *args_data)
        event.wait()

        return out_device


class SoftChopInterfaceToDevice:
    device_context = None
    device_queue = None

    device_program = None

    initialized = False

    def __init__(self, device_context, device_queue):
        if SoftChopInterfaceToDevice.initialized:
            return

        SoftChopInterfaceToDevice.device_context = device_context
        SoftChopInterfaceToDevice.device_queue = device_queue

        SoftChopInterfaceToDevice.device_program = cl.Program(device_context, softchop_source_code).build()

        SoftChopInterfaceToDevice.initialized = True

    @staticmethod
    def delta_back_prop(g_prime, new_delta, dz, out):
        device_global_shape = (int(np.prod(g_prime.shape)),)

        event = SoftChopInterfaceToDevice.device_program.delta_back_prop(SoftChopInterfaceToDevice.device_queue,
                                                                         device_global_shape, None,
                                                                         g_prime.data, new_delta.data, dz.data,
                                                                         out.data)
        event.wait()

    @staticmethod
    def parameter_gradient(delta, parameter, input_length, N, out):
        device_global_shape = (input_length,)
        event = SoftChopInterfaceToDevice.device_program.parameter_gradient(SoftChopInterfaceToDevice.device_queue,
                                                                            device_global_shape, None,
                                                                            delta.data, parameter.data,
                                                                            input_length, N, out.data)
        event.wait()


class SoftChop(NetworkNode, LayerBaseAttributes, Layer):
    def __init__(self, include_bias=True, weight_decay=0.0):
        NetworkNode.__init__(self)
        LayerBaseAttributes.__init__(self)

        self.include_bias = include_bias
        self.weight_decay = weight_decay

        self.a1 = None
        self.a2 = None
        self.epsilon1 = None
        self.epsilon2 = None
        self.bias = None

        self.b = None

    def build(self, device_context, device_queue):
        self.device_queue = device_queue
        self.device_context = device_context

        ClArrayTricks(device_context, device_queue)

        SoftChopInterfaceToDevice(self.device_context, self.device_queue)
        MultiSoftChop(self.device_context, self.device_queue)

        input_shape = self.parents[0].output_shape

        self.input_shape = input_shape
        self.output_shape = input_shape

        self.a1 = cl_array.to_device(device_queue, (np.random.rand(*self.input_shape) * 2).astype(np.float32))
        self.a2 = cl_array.to_device(device_queue, (np.random.rand(*self.input_shape) * 2).astype(np.float32))

        self.epsilon1 = cl_array.to_device(device_queue, (np.random.rand(*self.input_shape) * 2).astype(np.float32))
        self.epsilon2 = cl_array.to_device(device_queue, (np.random.rand(*self.input_shape) * 2).astype(np.float32))

        self.b = cl_array.zeros_like(self.a1)

        self.built = True

        self.clip_parameters()

    @check_built
    def clip_parameters(self, min_a=0.001, min_e=0.001):
        ClArrayTricks.clip_cl_array_in_place(self.a1, min_a, None)
        ClArrayTricks.clip_cl_array_in_place(self.a2, min_a, None)

        ClArrayTricks.clip_cl_array_in_place(self.epsilon1, min_e, None)
        ClArrayTricks.clip_cl_array_in_place(self.epsilon2, min_e, None)

    @check_built
    def predict(self, z, output_only=True, **kwargs):
        assert_instance_of_cl_array(z)

        out = MultiSoftChop.eval(z, self.a1, self.a2, self.epsilon1, self.epsilon2)

        if output_only:
            return out
        return out, out

    @check_built
    def get_delta_backprop_(self, g_prime, new_delta, prev_z):
        dz = MultiSoftChop.dx(prev_z, self.a1, self.a2, self.epsilon1, self.epsilon2)
        summed_delta_device = reduce(lambda x, y: x + y, new_delta)

        out_gpu = cl_array.empty_like(prev_z)
        SoftChopInterfaceToDevice.delta_back_prop(g_prime, summed_delta_device, dz, out_gpu)

        return out_gpu

    @check_built
    def get_parameter_gradients_(self, delta, prev_z, e=1e-7):
        args = (prev_z, self.a1, self.a2, self.epsilon1, self.epsilon2)

        dz = {'a1': MultiSoftChop.da1(*args),
              'a2': MultiSoftChop.da2(*args),
              'epsilon1': MultiSoftChop.de1(*args),
              'epsilon2': MultiSoftChop.de2(*args)}

        N = np.int32(len(prev_z))

        parameter_gradients = {'a1': cl_array.empty_like(self.a1),
                               'a2': cl_array.empty_like(self.a2),
                               'epsilon1': cl_array.empty_like(self.epsilon1),
                               'epsilon2': cl_array.empty_like(self.epsilon2)}

        summed_delta_device = reduce(lambda x, y: x + y, delta)

        for key in parameter_gradients.keys():
            SoftChopInterfaceToDevice.parameter_gradient(summed_delta_device, dz[key], self.input_length_device,
                                                         N, parameter_gradients[key])

        if abs(self.weight_decay) > e:
            parameter_gradients['a1'] += self.weight_decay * self.a1
            parameter_gradients['a2'] += self.weight_decay * self.a2

            parameter_gradients['epsilon1'] += self.weight_decay * self.epsilon1
            parameter_gradients['epsilon2'] += self.weight_decay * self.epsilon2

        return parameter_gradients

    @check_built
    def update_parameters_(self, parameter_updates):
        self.a1 -= parameter_updates['a1']
        self.a2 -= parameter_updates['a2']

        self.epsilon1 -= parameter_updates['epsilon1']
        self.epsilon2 -= parameter_updates['epsilon2']

        self.clip_parameters()

    @check_built
    def get_weights(self, as_dict=False):
        if as_dict:
            return {'a1': self.a1.get(),
                    'a2': self.a2.get(),
                    'epsilon1': self.epsilon1.get(),
                    'epsilon2': self.epsilon2.get(),
                    'b': self.b.get()}

        return (np.array([self.a1.get(), self.a2.get()]),
                np.array([self.epsilon1.get(), self.epsilon2.get()]),
                self.b.get())

    @check_built
    def summary_(self):
        return f'SoftChop {self.input_shape}', f'Output Shape {(None, *self.output_shape)}'

    def __str__(self):
        return f'SoftChop: Output Shape {(None, *self.output_shape)}'
