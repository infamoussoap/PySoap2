import numpy as np
from functools import reduce

import pyopencl.array as cl_array

from PySoap2_gpu.layers import Layer
from PySoap2_gpu.layers import NetworkNode
from PySoap2_gpu.layers.LayerBaseAttributes import LayerBaseAttributes

from .ValueChecks import assert_instance_of_cl_array
from .ValueChecks import check_built

from PySoap2_gpu.utils import ClArrayTricks
from PySoap2_gpu.utils import Broadcast

from PySoap2_gpu.layers.ProgramInterface.SoftChopInterface import SoftChopInterface, MultiSoftChopInterface


class SoftChop(NetworkNode, LayerBaseAttributes, Layer):
    """
        Notes
        -----
        weight_decay for the SoftChop maximises (not minimises) the values of the softchop parameters. This is because,
            - Larger a1/a2 values implies that more evidence is required to be non-zero
            - Larger epsilon1/epsilon2 values implies that more evidence is required to be important
    """

    def __init__(self, include_bias=True, weight_decay=0.0):
        NetworkNode.__init__(self)
        LayerBaseAttributes.__init__(self)

        self.include_bias = include_bias
        self.weight_decay = weight_decay

        self.a1 = None
        self.a2 = None
        self.epsilon1 = None
        self.epsilon2 = None
        self.b = None  # Bias unit

    def build(self, device_context, device_queue):
        self.queue = device_queue
        self.context = device_context

        ClArrayTricks(device_context, device_queue)
        Broadcast(device_context, device_queue)

        SoftChopInterface(self.context, self.queue)
        MultiSoftChopInterface(self.context, self.queue)

        input_shape = self.parents[0].output_shape

        self.input_shape = input_shape
        self.output_shape = input_shape

        self.a1 = cl_array.to_device(device_queue, (np.random.rand(*self.input_shape) * 2).astype(np.float64))
        self.a2 = cl_array.to_device(device_queue, (np.random.rand(*self.input_shape) * 2).astype(np.float64))

        self.epsilon1 = cl_array.to_device(device_queue, (np.random.rand(*self.input_shape) * 2).astype(np.float64))
        self.epsilon2 = cl_array.to_device(device_queue, (np.random.rand(*self.input_shape) * 2).astype(np.float64))

        if self.include_bias:
            self.b = cl_array.to_device(device_queue, np.random.rand(*self.input_shape).astype(np.float64))
        else:
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

        if self.include_bias:
            z = Broadcast.broadcast_across_0_axis('+', z, self.b)

        out = MultiSoftChopInterface.eval(z, self.a1, self.a2, self.epsilon1, self.epsilon2)

        if output_only:
            return out
        return out, out

    @check_built
    def get_delta_backprop_(self, g_prime, new_delta, prev_z):
        if self.include_bias:
            prev_z = Broadcast.broadcast_across_0_axis('+', prev_z, self.b)

        dz = MultiSoftChopInterface.dx(prev_z, self.a1, self.a2, self.epsilon1, self.epsilon2)
        summed_delta_device = reduce(lambda x, y: x + y, new_delta)

        out_gpu = cl_array.empty_like(prev_z)
        SoftChopInterface.delta_back_prop(g_prime, summed_delta_device, dz, out_gpu)

        return out_gpu

    @check_built
    def get_parameter_gradients_(self, delta, prev_z, e=1e-7):
        if self.include_bias:
            prev_z = Broadcast.broadcast_across_0_axis('+', prev_z, self.b)

        args = (prev_z, self.a1, self.a2, self.epsilon1, self.epsilon2)

        dz = {'a1': MultiSoftChopInterface.da1(*args),
              'a2': MultiSoftChopInterface.da2(*args),
              'epsilon1': MultiSoftChopInterface.de1(*args),
              'epsilon2': MultiSoftChopInterface.de2(*args)}

        if self.include_bias:
            dz['bias'] = MultiSoftChopInterface.dx(*args)

        N = np.int32(len(prev_z))

        parameter_gradients = {'a1': cl_array.empty_like(self.a1),
                               'a2': cl_array.empty_like(self.a2),
                               'epsilon1': cl_array.empty_like(self.epsilon1),
                               'epsilon2': cl_array.empty_like(self.epsilon2)}

        if self.include_bias:
            parameter_gradients['bias'] = cl_array.empty_like(self.b)

        summed_delta_device = reduce(lambda x, y: x + y, delta)

        for key in parameter_gradients.keys():
            SoftChopInterface.parameter_gradient(summed_delta_device, dz[key], self.input_length_device,
                                                 N, parameter_gradients[key])

        if abs(self.weight_decay) > e:
            parameter_gradients['a1'] -= self.weight_decay * self.a1
            parameter_gradients['a2'] -= self.weight_decay * self.a2

            parameter_gradients['epsilon1'] -= self.weight_decay * self.epsilon1
            parameter_gradients['epsilon2'] -= self.weight_decay * self.epsilon2

            if self.include_bias:
                parameter_gradients['bias'] = self.weight_decay * self.b

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
