from PySoap2.optimizers import SGD as CpuSGD
from .Optimizer import GPUOptimizer


class SGD(CpuSGD, GPUOptimizer):
    def __init__(self, learning_rate=0.001, momentum=0.0, nesterov=False):
        super().__init__(learning_rate=learning_rate, momentum=momentum, nesterov=nesterov)
