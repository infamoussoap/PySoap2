import pyopencl as cl

from ..c_code.polynomial_transformation_1d import polynomial_1d_transform_c_code
from ..c_code.polynomial_transformation_2d import polynomial_2d_transform_c_code


class PolynomialTransformationInterface:
    device_context = None
    device_queue = None

    device_program_for_1d = None
    device_program_for_2d = None

    initialize = False

    def __init__(self, context, queue):
        if PolynomialTransformationInterface.initialize:
            return

        PolynomialTransformationInterface.device_context = context
        PolynomialTransformationInterface.device_queue = queue

        PolynomialTransformationInterface.device_program_for_2d = cl.Program(context,
                                                                             polynomial_2d_transform_c_code).build()
        PolynomialTransformationInterface.device_program_for_1d = cl.Program(context,
                                                                             polynomial_1d_transform_c_code).build()

        PolynomialTransformationInterface.initialize = True

    @staticmethod
    def polynomial_transform_1d(P1, images, M1, input_length, out):
        """ images assumed to be (N, M1) cl_array.Array """
        program = PolynomialTransformationInterface.device_program_for_1d
        queue = PolynomialTransformationInterface.device_queue

        event = program.polynomial_transform_1d(queue, out.shape, None,
                                                P1.data, images.data, M1.data,
                                                input_length.data, out.data)
        event.wait()

    @staticmethod
    def polynomial_transform_1d_multi(P1, images, M1, M2, input_length, out):
        """ images assumed to be (N, M1, M2) cl_array.Array """
        program = PolynomialTransformationInterface.device_program_for_1d
        queue = PolynomialTransformationInterface.device_queue

        event = program.polynomial_transform_1d_multi(queue, images.shape, None,
                                                      P1.data, images.data, M1.data,
                                                      M2.data, input_length.data, out.data)
        event.wait()

    @staticmethod
    def polynomial_transform_2d(P1, P2, images, M1, M2, M3, input_length, out):
        if len(images.shape) == 3:
            PolynomialTransformationInterface._polynomial_transform_2d(P1, P2, images, M1, M2, M3, input_length, out)
        elif len(images.shape) == 4:
            PolynomialTransformationInterface._polynomial_transform_2d_multi(P1, P2, images, M1, M2, M3,
                                                                             input_length, out)
        else:
            raise ValueError(f'{images.shape} is not a valid shape for 2d transformation')

    @staticmethod
    def _polynomial_transform_2d(P1, P2, images, M1, M2, M3, input_length, out):
        """ images assumed to be (N, M1, M2) cl_array.Array """
        program = PolynomialTransformationInterface.device_program_for_2d
        queue = PolynomialTransformationInterface.device_queue

        event = program.polynomial_transform_2d(queue, out.shape, None,
                                                P1.data, P2.data, images.data, M1.data,
                                                M2.data, input_length.data, out.data)
        event.wait()

    @staticmethod
    def _polynomial_transform_2d_multi(P1, P2, images, M1, M2, M3, input_length, out):
        """ images assumed to be (N, M1, M2, M3) cl_array.Array """
        program = PolynomialTransformationInterface.device_program_for_2d
        queue = PolynomialTransformationInterface.device_queue

        # Global shape can only be up to 3-d
        # Combining the last 2 axis as 1 solves this problem
        n, m1, m2, m3 = images.shape
        global_shape = (n, m1, m2 * m3)

        event = program.polynomial_transform_2d_multi(queue, global_shape, None,
                                                      P1.data, P2.data, images.data, M1.data,
                                                      M2.data, M3.data, input_length.data, out.data)
        event.wait()
