import numpy as np
from functools import reduce

import pyopencl as cl
import pyopencl.array as cl_array

from PySoap2_gpu.layers import Layer
from PySoap2_gpu.layers import NetworkNode
from PySoap2_gpu.layers.LayerBaseAttributes import LayerBaseAttributes

from .c_code.conv2d_c_code import conv2d_source_code
from .ValueChecks import assert_instance_of_cl_array
from .ValueChecks import check_built

from PySoap2_gpu.Exceptions import check_for_valid_context
from PySoap2_gpu.utils import ClArrayTricks


class Conv2DInterfaceToDevice:
    device_context = None
    device_queue = None

    device_program = None

    initialized = False

    def __init__(self, device_context, device_queue):
        """ Compile the c-program

            Notes
            -----
            Once this class has been initialized, the c-program will be compiled on the given device context and
            will be bound to the class (not instances of the class).
            It will no longer be possible to re-initialize this class again.
        """
        if Conv2DInterfaceToDevice.initialized:
            return

        Conv2DInterfaceToDevice.device_context = device_context
        Conv2DInterfaceToDevice.device_queue = device_queue

        Conv2DInterfaceToDevice.device_program = cl.Program(device_context, conv2d_source_code).build()

        Conv2DInterfaceToDevice.initialized = True

    @staticmethod
    def predict(z, filter_, bias_,
                filter_height, filter_width, num_of_filters,
                stride,
                image_width, image_depth,
                output_width,
                input_length, output_length,
                out):

        output_height = int(output_length / (output_width * num_of_filters))

        program = Conv2DInterfaceToDevice.device_program
        queue = Conv2DInterfaceToDevice.device_queue

        events = []
        for i in range(num_of_filters):
            current_filter = np.int32(i)

            event = program.predict(queue, (len(z), output_height, output_width), None,
                                    z.data, filter_.data, bias_.data,
                                    filter_height, filter_width, num_of_filters,
                                    stride, current_filter,
                                    image_width, image_depth,
                                    output_width,
                                    input_length, output_length,
                                    out.data)

            events.append(event)

        cl.wait_for_events(events)
        # queue.finish()

    @staticmethod
    def delta_back_prop(delta, eye_conv, g_prime, input_length, output_length, out):
        program = Conv2DInterfaceToDevice.device_program
        queue = Conv2DInterfaceToDevice.device_queue

        global_shape = (np.prod(out.shape),)

        event = program.delta_back_prop(queue, global_shape, None,
                                        delta.data, eye_conv.data, g_prime.data, input_length, output_length, out.data)

        event.wait()
        return out

    @staticmethod
    def filter_gradient(prev_z, delta,
                        output_height, output_width, num_of_filters,
                        stride,
                        image_width, image_depth,
                        N, input_length,
                        filter_width,
                        out):
        program = Conv2DInterfaceToDevice.device_program
        queue = Conv2DInterfaceToDevice.device_queue

        global_shape = (np.prod(out.shape),)

        event = program.filter_gradient(queue, global_shape, None,
                                        prev_z.data, delta.data,
                                        output_height, output_width, num_of_filters,
                                        stride,
                                        image_width, image_depth,
                                        N, input_length,
                                        filter_width,
                                        out.data)
        event.wait()

    @staticmethod
    def bias_gradient(delta, sum_length, num_of_filters, out):
        program = Conv2DInterfaceToDevice.device_program
        queue = Conv2DInterfaceToDevice.device_queue

        global_shape = out.shape

        event = program.bias_gradient(queue, global_shape, None,
                                      delta.data, sum_length, num_of_filters, out.data)
        event.wait()


class Conv2D(NetworkNode, LayerBaseAttributes, Layer):

    def __init__(self, filter_num, filter_spatial_shape, stride, activation_function,
                 activation_kwargs=None, padding="VALID"):
        NetworkNode.__init__(self)
        LayerBaseAttributes.__init__(self)

        self.filter_num = np.int32(filter_num)
        self.filter_spatial_shape = filter_spatial_shape

        self.stride = np.int32(stride)

        self.activation_function = activation_function
        self.activation_kwargs = {} if activation_kwargs is None else activation_kwargs

        self.single_filter_shape = None
        self.filter_shape = None
        self.output_spatial_shape = None

        self.filter = None
        self.b = None

        if padding in ["VALID", "SAME"]:
            self.padding = padding
        else:
            raise ValueError(f"Padding {padding} is invalid. Try 'VALID' or 'SAME' padding.")

    def build(self, device_context, device_queue):
        self.device_queue = device_queue
        self.device_context = device_context

        Conv2DInterfaceToDevice(self.device_context, self.device_queue)
        ClArrayTricks(device_context, device_queue)

        input_shape = self.parents[0].output_shape

        self.input_shape = input_shape
        self.single_filter_shape = (*self.filter_spatial_shape, input_shape[2])

        self.filter_shape = (*self.single_filter_shape, self.filter_num)

        if self.padding == "VALID":
            # These 2 lines follow the formula in the youtube lecture
            # Giving us the output shape of this layer
            n = int((input_shape[0] - self.filter_spatial_shape[0]) / self.stride + 1)
            m = int((input_shape[1] - self.filter_spatial_shape[1]) / self.stride + 1)
        else:  # padding == "SAME"
            n = input_shape[0]
            m = input_shape[1]

        self.output_spatial_shape = (n, m)
        self.output_shape = (*self.output_spatial_shape, self.filter_num)

        # Initialise the the filter with Glorot-Uniform, a uniform distribution over [-limit, limit],
        # where limit = sqrt(6 / (fan_in + fan_out)) (fan_in is the number of input units in the weight
        # tensor and fan_out is the number of output units).
        limit = np.sqrt(6 / (np.prod(self.filter_shape) + 1))
        filter_ = np.random.uniform(low=-limit, high=limit, size=self.filter_shape)
        b = np.zeros(self.filter_num)

        self.filter = cl_array.to_device(self.device_queue, filter_.astype(np.float64))
        self.b = cl_array.to_device(self.device_queue, b.astype(np.float64))

        self.built = True

    @check_built
    def predict(self, z, output_only=True, **kwargs):
        z = self.pad_images(z, self.padding)
        out = Conv2D.perform_conv(z, self.filter, self.b, self.stride, self.device_queue)

        if output_only:
            return self.activation_function_(out)
        return out, self.activation_function_(out)

    @check_built
    def get_delta_backprop_(self, g_prime, new_delta, *args):
        assert_instance_of_cl_array(g_prime)

        delta = reduce(lambda x, y: x + y, new_delta)
        padded_delta = self.pad_images(delta, "FULL")

        flipped_filter = ClArrayTricks.flip_across_0_1_axis(self.filter)
        flipped_filter = ClArrayTricks.transpose_last_two_axis(flipped_filter)
        b = cl_array.zeros(self.device_queue, flipped_filter.shape[-1], np.float64)

        temp1 = np.flip(self.filter.get(), axis=(0, 1)).transpose((0, 1, 3, 2))
        temp2 = flipped_filter.get()
        np.testing.assert_almost_equal(temp1, temp2)

        ds_dz = self.perform_conv(padded_delta, flipped_filter, b, self.stride, self.device_queue)
        ds_dz = self.remove_pad(ds_dz, self.padding)

        return ds_dz * g_prime

    @check_built
    def get_parameter_gradients_(self, delta, prev_z):
        assert_instance_of_cl_array(prev_z)
        delta = reduce(lambda x, y: x + y, delta)

        prev_z = self.pad_images(prev_z, self.padding)

        filter_grads = cl_array.zeros(self.device_queue, self.filter.shape, dtype=np.float64)
        bias_grads = cl_array.zeros(self.device_queue, self.b.shape, dtype=np.float64)

        output_height, output_width = self.output_spatial_shape
        _, image_width, image_depth = self.input_shape
        _, filter_width, _ = self.single_filter_shape
        N = len(prev_z)

        Conv2DInterfaceToDevice.filter_gradient(prev_z, delta,
                                                np.int32(output_height), np.int32(output_width), self.filter_num,
                                                np.int32(self.stride),
                                                np.int32(image_width), np.int32(image_depth),
                                                np.int32(N), self.input_length_device,
                                                np.int32(filter_width),
                                                filter_grads)

        sum_length = np.int32(np.prod(delta.shape[:-1]))
        Conv2DInterfaceToDevice.bias_gradient(delta, sum_length, self.filter_num, bias_grads)

        return {'filter': filter_grads, 'bias': bias_grads}

    @check_built
    def update_parameters_(self, parameter_updates):
        self.filter -= parameter_updates['filter']
        self.b -= parameter_updates['bias']

    @check_built
    def get_weights(self, as_dict=False):
        if as_dict:
            return {'filter': self.filter.get(), 'b': self.b.get()}
        return self.filter.get(), self.b.get()

    @check_built
    def summary_(self):
        return f'Conv2D {self.filter_num} x {(self.single_filter_shape,)}', f'Output Shape {(None, *self.output_shape)}'

    @staticmethod
    def perform_conv(images, filter_, bias, stride, device_queue):
        assert_instance_of_cl_array(images)

        input_shape = images.shape[1:]
        n = int((input_shape[0] - filter_.shape[0]) / stride + 1)
        m = int((input_shape[1] - filter_.shape[1]) / stride + 1)
        output_shape = (n, m, filter_.shape[3])

        out = cl_array.zeros(device_queue, (len(images), *output_shape), dtype=np.float64)

        filter_height, filter_width, _, filter_num = filter_.shape
        _, image_width, image_depth = input_shape
        output_width = np.int32(output_shape[1])
        input_length, output_length = np.int32(np.prod(input_shape)), np.int32(np.prod(output_shape))

        Conv2DInterfaceToDevice.predict(images, filter_, bias,
                                        np.int32(filter_height), np.int32(filter_width), np.int32(filter_num),
                                        np.int32(stride),
                                        np.int32(image_width), np.int32(image_depth),
                                        output_width,
                                        input_length, output_length,
                                        out)
        return out

    def pad_images(self, images, padding):
        if padding == "VALID":
            return images

        elif padding == "SAME":
            height_pad_length = self.input_shape[0] - 1 - int(
                (self.input_shape[0] - self.filter_shape[0]) / self.stride)
            width_pad_length = self.input_shape[1] - 1 - int((self.input_shape[1] - self.filter_shape[1]) / self.stride)

            upper_pad = height_pad_length // 2
            lower_pad = height_pad_length - upper_pad

            left_pad = width_pad_length // 2
            right_pad = width_pad_length - left_pad
        elif padding == "FULL":
            filter_row, filter_col = self.filter_spatial_shape

            upper_pad = filter_row - 1
            lower_pad = filter_row - 1

            left_pad = filter_col - 1
            right_pad = filter_col - 1

        else:
            raise ValueError(f"Padding {padding} is invalid.")

        padded_images = ClArrayTricks.pad_images(images, upper_pad, lower_pad, left_pad, right_pad)
        return padded_images

    def remove_pad(self, images, padding):
        if padding == "VALID":
            return images

        elif padding == "SAME":
            height_pad_length = self.input_shape[0] - 1 - int(
                (self.input_shape[0] - self.filter_shape[0]) / self.stride)
            width_pad_length = self.input_shape[1] - 1 - int((self.input_shape[1] - self.filter_shape[1]) / self.stride)

            upper_pad = height_pad_length // 2
            lower_pad = height_pad_length - upper_pad

            left_pad = width_pad_length // 2
            right_pad = width_pad_length - left_pad
        elif padding == "FULL":
            filter_row, filter_col = self.filter_spatial_shape

            upper_pad = filter_row - 1
            lower_pad = filter_row - 1

            left_pad = filter_col - 1
            right_pad = filter_col - 1

        else:
            raise ValueError(f"Padding {padding} is invalid.")

        return ClArrayTricks.remove_pad(images, upper_pad, lower_pad, left_pad, right_pad)
