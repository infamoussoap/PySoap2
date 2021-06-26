from .LearningRateScheduler import LearningRateScheduler


class ConstantLearningRate(LearningRateScheduler):
    def __init__(self, lr):
        self.lr = lr

    def get(self):
        return self.lr
