import pyopencl as cl

from .c_code.multi_softchop_c_code import multi_softchop_source_code


class MultiSoftChop:
    def __init__(self, gpu_context, gpu_queue):
        self.gpu_context = gpu_context
        self.gpu_queue = gpu_queue

        self.gpu_program = cl.Program(self.gpu_context, multi_softchop_source_code).build()
