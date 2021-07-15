import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array


class BroadcastError(Exception):
    pass


class Broadcast:
    device_context = None
    device_queue = None

    device_program = None

    initialized = False

    def __init__(self, context, queue):
        # If this class is initialized, it means that the programs is already on the device
        if Broadcast.initialized:
            return
        Broadcast.device_context = context
        Broadcast.device_queue = queue

        Broadcast.device_program = cl.Program(context, broadcast_across_axis_c_code).build()

        Broadcast.initialized = True

    @staticmethod
    def broadcast_across_0_axis(operation, x_device, y_device):
        """ It is assumed that the operation to be performed is

            x_device 'operation' y_device

            If operation = '-', then this will return x_device - y_device
        """

        error_msg = f'Broadcasting two arrays must the last dimension equal. ' \
                    f'In particular, x.shape={x_device.shape} not compatible y.shape={y_device.shape}.'
        try:
            return Broadcast._broadcast_across_0_axis(operation, x_device, y_device, error_msg)
        except BroadcastError:
            """ Now assumed that 
            
                y_device : (N, ...) cl_array.Array
                x_device : (...) cl_array.Array
            """
            if operation == "-":  # These are special operation that is sensitive on the position
                return Broadcast._broadcast_across_0_axis('+', -y_device, x_device, error_msg)
            elif operation == '/':
                # Special divide 
                return Broadcast._broadcast_across_0_axis('_/', y_device, x_device, error_msg)
            else:
                return Broadcast._broadcast_across_0_axis(operation, y_device, x_device, error_msg)

    @staticmethod
    def _broadcast_across_0_axis(operation, x_device, y_device, error_msg):
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
            raise BroadcastError(error_msg)

        input_shape = tuple(input_shape)
        input_length = np.int32(np.prod(input_shape))
        N = np.int32(N)

        out = cl_array.empty_like(x_device)

        broadcast_program = Broadcast._get_broadcast_program(operation)
        event = broadcast_program(Broadcast.device_queue, (N, input_length), None,
                                  x_device.data, y_device.data, input_length, N, out.data)
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
        elif operation == "_/":
            return Broadcast.device_program._broadcast_div_across_0_axis
        else:
            raise ValueError(f"{operation} is not a valid operation. Choose from +,-,*,/.")


broadcast_across_axis_c_code = """
__kernel void broadcast_add_across_0_axis(__global const float *x, __global const float *y, const int input_length, 
                                          const int N, __global float *out)
{
    int i = get_global_id(0);
    int j = get_global_id(1);

    out[j + i*input_length] = x[j + i*input_length] + y[j];
}

__kernel void broadcast_sub_across_0_axis(__global const float *x, __global const float *y, const int input_length, 
                                          const int N, __global float *out)
{
    int i = get_global_id(0);
    int j = get_global_id(1);

    out[j + i*input_length] = x[j + i*input_length] - y[j];
}

__kernel void broadcast_mul_across_0_axis(__global const float *x, __global const float *y, const int input_length, 
                                          const int N, __global float *out)
{
    int i = get_global_id(0);
    int j = get_global_id(1);

    out[j + i*input_length] = (float) ((double) x[j + i*input_length] * (double) y[j]);
}

__kernel void broadcast_div_across_0_axis(__global const float *x, __global const float *y, const int input_length, 
                                          const int N, __global float *out)
{
    int i = get_global_id(0);
    int j = get_global_id(1);

    out[j + i*input_length] = (float) ((double) x[j + i*input_length] / (double) y[j]);
}

__kernel void _broadcast_div_across_0_axis(__global const float *x, __global const float *y, const int input_length, 
                                           const int N, __global float *out)
{
    int i = get_global_id(0);
    int j = get_global_id(1);

    out[j + i*input_length] = (float) ((double) y[j]) / (double) x[j + i*input_length];
}
"""
