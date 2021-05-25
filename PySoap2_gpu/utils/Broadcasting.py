import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array


class Broadcast:
    device_context = None
    device_queue = None

    device_program = None

    initialized = False

    def __init__(self, context, queue):
        Broadcast.device_context = context
        Broadcast.device_queue = queue

        Broadcast.device_program = cl.Program(context, broadcast_across_axis_c_code).build()

        Broadcast.initialized = True

    @staticmethod
    def broadcast_across_0_axis(operation, x_device, y_device):
        """ Performs broadcasting on the gpu

            Parameters
            ----------
            x_device : (N, ...) cl_array.Array
            y_device : (...) cl_array.Array
            operation : str
                The operation to be performed. Only +,-,*,/ are supported

            y is to be broadcast against x
        """
        N, *input_shape = x_device.shape

        if tuple(input_shape) != y_device.shape:
            raise ValueError(f'Broadcasting two arrays must the last dimension equal. '
                             f'In particular, x.shape={x_device.shape} not equal y.shape={y_device.shape}.')

        input_shape = tuple(input_shape)
        input_length = int(np.prod(input_shape))

        input_length_device = cl_array.to_device(Broadcast.device_queue, np.array(input_length, dtype=np.int32))
        N_device = cl_array.to_device(Broadcast.device_queue, np.array(N, dtype=np.int32))
        out = cl_array.empty_like(x_device)

        broadcast_program = Broadcast._get_broadcast_program(operation)
        event = broadcast_program(Broadcast.device_queue, (N, input_length), None,
                                  x_device.data, y_device.data, input_length_device.data,
                                  N_device.data, out.data)
        event.wait()
        return out

    @staticmethod
    def _get_broadcast_program(operation):
        if operation == "+":
            return Broadcast.device_program.broadcast_add_across_0_axis
        elif operation == "-":
            return Broadcast.device_program.broadcast_sub_across_0_axis
        elif operation == "*":
            return Broadcast.device_program.broadcast_mul_across_0_axis
        elif operation == "/":
            return Broadcast.device_program.broadcast_div_across_0_axis
        else:
            raise ValueError(f"{operation} is not a valid operation. Choose from +,-,*,/.")


broadcast_across_axis_c_code = """
__kernel void broadcast_add_across_0_axis(__global const float *x, __global const float *y, __global int *inputLength, 
                                          __global int *N_, __global float *out)
{
    int i = get_global_id(0);
    int j = get_global_id(1);

    int input_length = *inputLength;
    int N = *N_;

    float total = 0.0;

    out[j + i*input_length] = x[j + i*input_length] + y[j];
}

__kernel void broadcast_sub_across_0_axis(__global const float *x, __global const float *y, __global int *inputLength, 
                                          __global int *N_, __global float *out)
{
    int i = get_global_id(0);
    int j = get_global_id(1);

    int input_length = *inputLength;
    int N = *N_;

    float total = 0.0;

    out[j + i*input_length] = x[j + i*input_length] - y[j];
}

__kernel void broadcast_mul_across_0_axis(__global const float *x, __global const float *y, __global int *inputLength, 
                                          __global int *N_, __global float *out)
{
    int i = get_global_id(0);
    int j = get_global_id(1);

    int input_length = *inputLength;
    int N = *N_;

    float total = 0.0;

    out[j + i*input_length] = x[j + i*input_length] * y[j];
}

__kernel void broadcast_div_across_0_axis(__global const float *x, __global const float *y, __global int *inputLength, 
                                          __global int *N_, __global float *out)
{
    int i = get_global_id(0);
    int j = get_global_id(1);

    int input_length = *inputLength;
    int N = *N_;

    float total = 0.0;

    out[j + i*input_length] = x[j + i*input_length] / y[j];
}
"""
