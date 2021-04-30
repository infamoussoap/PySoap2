import numpy as np

import pyopencl as cl
import pyopencl.array as cl_array

from .c_code.multi_softchop_c_code import multi_softchop_source_code
from .c_code.softchop_c_code import softchop_source_code

from PySoap2_gpu.layers import Layer
from PySoap2_gpu.layers.NetworkNode import NetworkNode
from PySoap2_gpu.layers.LayerBaseAttributes import LayerBaseAttributes

from .ValueChecks import assert_instance_of_cl_array


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

        self.gpu_program.softchop_eval(self.gpu_queue, (N, input_length), None, *args_data)

        return out_device

    def dx(self, x_device, a1_device, a2_device, epsilon1_device, epsilon2_device):
        out_device = cl_array.empty_like(x_device)

        input_length = np.array(np.prod(a1_device.shape)).astype(np.int32)
        N = np.array(len(x_device)).astype(np.int32)

        input_length_device = cl_array.to_device(self.gpu_queue, input_length)

        args = [x_device, a1_device, a2_device, epsilon1_device, epsilon2_device, input_length_device, out_device]
        args_data = [arg.data for arg in args]

        self.gpu_program.softchop_dx(self.gpu_queue, (N, input_length), None, *args_data)

        return out_device

    def da1(self, x_device, a1_device, a2_device, epsilon1_device, epsilon2_device):
        out_device = cl_array.empty_like(x_device)

        input_length = np.array(np.prod(a1_device.shape)).astype(np.int32)
        N = np.array(len(x_device)).astype(np.int32)

        input_length_device = cl_array.to_device(self.gpu_queue, input_length)

        args = [x_device, a1_device, a2_device, epsilon1_device, epsilon2_device, input_length_device, out_device]
        args_data = [arg.data for arg in args]

        self.gpu_program.softchop_da1(self.gpu_queue, (N, input_length), None, *args_data)

        return out_device

    def da2(self, x_device, a1_device, a2_device, epsilon1_device, epsilon2_device):
        out_device = cl_array.empty_like(x_device)

        input_length = np.array(np.prod(a1_device.shape)).astype(np.int32)
        N = np.array(len(x_device)).astype(np.int32)

        input_length_device = cl_array.to_device(self.gpu_queue, input_length)

        args = [x_device, a1_device, a2_device, epsilon1_device, epsilon2_device, input_length_device, out_device]
        args_data = [arg.data for arg in args]

        self.gpu_program.softchop_da2(self.gpu_queue, (N, input_length), None, *args_data)

        return out_device

    def de1(self, x_device, a1_device, a2_device, epsilon1_device, epsilon2_device):
        out_device = cl_array.empty_like(x_device)

        input_length = np.array(np.prod(a1_device.shape)).astype(np.int32)
        N = np.array(len(x_device)).astype(np.int32)

        input_length_device = cl_array.to_device(self.gpu_queue, input_length)

        args = [x_device, a1_device, a2_device, epsilon1_device, epsilon2_device, input_length_device, out_device]
        args_data = [arg.data for arg in args]

        self.gpu_program.softchop_de1(self.gpu_queue, (N, input_length), None, *args_data)

        return out_device

    def de2(self, x_device, a1_device, a2_device, epsilon1_device, epsilon2_device):
        out_device = cl_array.empty_like(x_device)

        input_length = np.array(np.prod(a1_device.shape)).astype(np.int32)
        N = np.array(len(x_device)).astype(np.int32)

        input_length_device = cl_array.to_device(self.gpu_queue, input_length)

        args = [x_device, a1_device, a2_device, epsilon1_device, epsilon2_device, input_length_device, out_device]
        args_data = [arg.data for arg in args]

        self.gpu_program.softchop_de2(self.gpu_queue, (N, input_length), None, *args_data)

        return out_device


class SoftChop(NetworkNode, LayerBaseAttributes, Layer):
    def __init__(self, include_bias=True):
        NetworkNode.__init__(self)
        LayerBaseAttributes.__init__(self)

        self.include_bias = include_bias

        self.a1 = None
        self.a2 = None
        self.e1 = None
        self.e2 = None

        self.b = None

    def build(self, gpu_context, gpu_queue):
        self.gpu_queue = gpu_queue
        self.gpu_context = gpu_context

        self.gpu_program = cl.Program(self.gpu_context, softchop_source_code).build()

        input_shape = self.parents[0].output_shape

        self.input_shape = input_shape
        self.output_shape = input_shape

        self.a1 = cl_array.to_device(gpu_queue, (np.random.rand(*self.input_shape) * 2).astype(np.float32))
        self.a2 = cl_array.to_device(gpu_queue, (np.random.rand(*self.input_shape) * 2).astype(np.float32))

        self.e1 = cl_array.to_device(gpu_queue, (np.random.rand(*self.input_shape) * 2).astype(np.float32))
        self.e2 = cl_array.to_device(gpu_queue, (np.random.rand(*self.input_shape) * 2).astype(np.float32))

        self.MultiSoftChop = MultiSoftChop(self.gpu_context, self.gpu_queue)
        self.clip_cl_array_in_place = cl.elementwise.ElementwiseKernel(gpu_context,
                                                                       "float *x, float threshold",
                                                                       "x[i] = x[i] > threshold ? x[i] : threshold",
                                                                       "clip_in_place_elementwise")

        self.built = True

        self.clip_parameters()

    def clip_parameters(self, min_a=0.001, min_e=0.001):
        self.clip_cl_array_in_place(self.a1)
        self.clip_cl_array_in_place(self.a2)

        self.clip_cl_array_in_place(self.e1)
        self.clip_cl_array_in_place(self.e2)

    def predict(self, z, output_only=True, **kwargs):
        assert_instance_of_cl_array(z)

        out = self.MultiSoftChop.eval(z, self.a1, self.a2, self.e1, self.e2)

        if output_only:
            return out
        return out, out

    def get_delta_backprop_(self, g_prime, new_delta, prev_z):
        out_gpu = cl_array.empty_like(prev_z)

        dz = self.MultiSoftChop.dx(prev_z, self.a1, self.a2, self.e1, self.e2)

        args = [g_prime, new_delta, dz, out_gpu]
        args_data = [arg.data for arg in args]

        self.gpu_program.delta_back_prop(self.gpu_queue, int(np.prod(g_prime.shape)), None, *args_data)

    def get_parameter_gradients_(self, delta, prev_z):
        args = (prev_z, self.a1, self.a2, self.e1, self.e2)

        dz = {'da1': self.MultiSoftChop.da1(*args),
              'da2': self.MultiSoftChop.da2(*args),
              'de1': self.MultiSoftChop.de1(*args),
              'de2': self.MultiSoftChop.de2(*args)}

        self.gpu_program.parameter_gradient(self.gpu_queue, )

        parameter_gradients = {}
