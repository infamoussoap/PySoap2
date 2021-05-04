import numpy as np

import pyopencl as cl
import pyopencl.array as cl_array

from .c_code.multi_softchop_c_code import multi_softchop_source_code
from .c_code.softchop_c_code import softchop_source_code

from PySoap2_gpu.layers import Layer
from PySoap2_gpu.layers.NetworkNode import NetworkNode
from PySoap2_gpu.layers.LayerBaseAttributes import LayerBaseAttributes

from .ValueChecks import assert_instance_of_cl_array

from PySoap2_gpu.utils import ClArrayTricks


class MultiSoftChop:
    def __init__(self, gpu_context, gpu_queue):
        self.gpu_context = gpu_context
        self.gpu_queue = gpu_queue

        self.gpu_program = cl.Program(self.gpu_context, multi_softchop_source_code).build()

    def eval(self, x_device, a1_device, a2_device, epsilon1_device, epsilon2_device):
        out_device = cl_array.empty_like(x_device)

        input_length = np.array(np.prod(a1_device.shape)).astype(np.int32)
        N = np.array(len(x_device)).astype(np.int32)

        input_length_device = cl_array.to_device(self.gpu_queue, input_length)

        args = [x_device, a1_device, a2_device, epsilon1_device, epsilon2_device, input_length_device, out_device]
        args_data = [arg.data for arg in args]

        event = self.gpu_program.softchop_eval(self.gpu_queue, (N, input_length), None, *args_data)
        event.wait()

        return out_device

    def dx(self, x_device, a1_device, a2_device, epsilon1_device, epsilon2_device):
        out_device = cl_array.empty_like(x_device)

        input_length = np.array(np.prod(a1_device.shape)).astype(np.int32)
        N = np.array(len(x_device)).astype(np.int32)

        input_length_device = cl_array.to_device(self.gpu_queue, input_length)

        args = [x_device, a1_device, a2_device, epsilon1_device, epsilon2_device, input_length_device, out_device]
        args_data = [arg.data for arg in args]

        event = self.gpu_program.softchop_dx(self.gpu_queue, (N, input_length), None, *args_data)
        event.wait()

        return out_device

    def da1(self, x_device, a1_device, a2_device, epsilon1_device, epsilon2_device):
        out_device = cl_array.empty_like(x_device)

        input_length = np.array(np.prod(a1_device.shape)).astype(np.int32)
        N = np.array(len(x_device)).astype(np.int32)

        input_length_device = cl_array.to_device(self.gpu_queue, input_length)

        args = [x_device, a1_device, a2_device, epsilon1_device, epsilon2_device, input_length_device, out_device]
        args_data = [arg.data for arg in args]

        event = self.gpu_program.softchop_da1(self.gpu_queue, (N, input_length), None, *args_data)
        event.wait()

        return out_device

    def da2(self, x_device, a1_device, a2_device, epsilon1_device, epsilon2_device):
        out_device = cl_array.empty_like(x_device)

        input_length = np.array(np.prod(a1_device.shape)).astype(np.int32)
        N = np.array(len(x_device)).astype(np.int32)

        input_length_device = cl_array.to_device(self.gpu_queue, input_length)

        args = [x_device, a1_device, a2_device, epsilon1_device, epsilon2_device, input_length_device, out_device]
        args_data = [arg.data for arg in args]

        event = self.gpu_program.softchop_da2(self.gpu_queue, (N, input_length), None, *args_data)
        event.wait()

        return out_device

    def de1(self, x_device, a1_device, a2_device, epsilon1_device, epsilon2_device):
        out_device = cl_array.empty_like(x_device)

        input_length = np.array(np.prod(a1_device.shape)).astype(np.int32)
        N = np.array(len(x_device)).astype(np.int32)

        input_length_device = cl_array.to_device(self.gpu_queue, input_length)

        args = [x_device, a1_device, a2_device, epsilon1_device, epsilon2_device, input_length_device, out_device]
        args_data = [arg.data for arg in args]

        event = self.gpu_program.softchop_de1(self.gpu_queue, (N, input_length), None, *args_data)
        event.wait()

        return out_device

    def de2(self, x_device, a1_device, a2_device, epsilon1_device, epsilon2_device):
        out_device = cl_array.empty_like(x_device)

        input_length = np.array(np.prod(a1_device.shape)).astype(np.int32)
        N = np.array(len(x_device)).astype(np.int32)

        input_length_device = cl_array.to_device(self.gpu_queue, input_length)

        args = [x_device, a1_device, a2_device, epsilon1_device, epsilon2_device, input_length_device, out_device]
        args_data = [arg.data for arg in args]

        event = self.gpu_program.softchop_de2(self.gpu_queue, (N, input_length), None, *args_data)
        event.wait()

        return out_device


class SoftChopInterfaceToDevice:
    def __init__(self, device_context, device_queue):
        self.device_queue = device_queue

        self.device_program = cl.Program(device_context, softchop_source_code).build()

    def delta_back_prop(self, g_prime, new_delta, dz, out):
        device_global_shape = (int(np.prod(g_prime.shape)),)

        event = self.device_program.delta_back_prop(self.device_queue, device_global_shape, None,
                                                    g_prime.data, new_delta.data, dz.data, out.data)
        event.wait()

    def parameter_gradient(self, delta, parameter, input_length, N, out):
        device_global_shape = (input_length.get(),)
        event = self.device_program.parameter_gradient(self.device_queue, device_global_shape, None,
                                                       delta.data, parameter.data, input_length.data, N.data, out.data)
        event.wait()


class SoftChop(SoftChopInterfaceToDevice, NetworkNode, LayerBaseAttributes, Layer):
    def __init__(self, include_bias=True):
        NetworkNode.__init__(self)
        LayerBaseAttributes.__init__(self)

        self.include_bias = include_bias

        self.a1 = None
        self.a2 = None
        self.e1 = None
        self.e2 = None

        self.b = None

    def build(self, device_context, device_queue):
        self.device_queue = device_queue
        self.device_context = device_context

        if not ClArrayTricks.initialized:
            ClArrayTricks(device_context, device_queue)

        SoftChopInterfaceToDevice.__init__(self, self.device_context, self.device_queue)

        input_shape = self.parents[0].output_shape

        self.input_shape = input_shape
        self.output_shape = input_shape

        self.input_length_device = cl_array.to_device(self.device_queue, np.array(np.prod(self.input_shape),
                                                                                  dtype=np.int32))
        self.output_length_device = cl_array.to_device(self.device_queue, np.array(np.prod(self.output_shape),
                                                                                   dtype=np.int32))

        self.a1 = cl_array.to_device(device_queue, (np.random.rand(*self.input_shape) * 2).astype(np.float32))
        self.a2 = cl_array.to_device(device_queue, (np.random.rand(*self.input_shape) * 2).astype(np.float32))

        self.e1 = cl_array.to_device(device_queue, (np.random.rand(*self.input_shape) * 2).astype(np.float32))
        self.e2 = cl_array.to_device(device_queue, (np.random.rand(*self.input_shape) * 2).astype(np.float32))

        self.MultiSoftChop = MultiSoftChop(self.device_context, self.device_queue)

        self.built = True

        self.clip_parameters()

    def clip_parameters(self, min_a=0.001, min_e=0.001):
        ClArrayTricks.clip_cl_array_in_place(self.a1, min_a, None)
        ClArrayTricks.clip_cl_array_in_place(self.a2, min_a, None)

        ClArrayTricks.clip_cl_array_in_place(self.e1, min_e, None)
        ClArrayTricks.clip_cl_array_in_place(self.e2, min_e, None)

    def predict(self, z, output_only=True, **kwargs):
        assert_instance_of_cl_array(z)

        out = self.MultiSoftChop.eval(z, self.a1, self.a2, self.e1, self.e2)

        if output_only:
            return out
        return out, out

    def get_delta_backprop_(self, g_prime, new_delta, prev_z):
        dz = self.MultiSoftChop.dx(prev_z, self.a1, self.a2, self.e1, self.e2)

        out_gpu = cl_array.empty_like(prev_z)
        super().delta_back_prop(g_prime, new_delta, dz, out_gpu)

        return out_gpu

    def get_parameter_gradients_(self, delta, prev_z):
        args = (prev_z, self.a1, self.a2, self.e1, self.e2)

        dz = {'a1': self.MultiSoftChop.da1(*args),
              'a2': self.MultiSoftChop.da2(*args),
              'e1': self.MultiSoftChop.de1(*args),
              'e2': self.MultiSoftChop.de2(*args)}

        N = np.array(len(prev_z)).astype(np.int32)
        N_device = cl_array.to_device(self.device_queue, N)

        parameter_gradients = {'a1': cl_array.empty_like(self.a1),
                               'a2': cl_array.empty_like(self.a2),
                               'e1': cl_array.empty_like(self.e1),
                               'e2': cl_array.empty_like(self.e2)}

        for key in parameter_gradients.keys():
            super().parameter_gradient(delta, dz[key], self.input_length_device, N_device, parameter_gradients[key])

        return parameter_gradients

    def update_parameters_(self, parameter_updates):
        self.a1 -= parameter_updates['a1']
        self.a2 -= parameter_updates['a2']

        self.e1 -= parameter_updates['e1']
        self.e2 -= parameter_updates['e2']

        self.clip_parameters()

    def get_weights(self):
        raise NotImplementedError

    def summary_(self):
        return f'SoftChop {self.input_shape}', f'Output Shape {(None, *self.output_shape)}'

    def __str__(self):
        return f'SoftChop: Output Shape {(None, *self.output_shape)}'
