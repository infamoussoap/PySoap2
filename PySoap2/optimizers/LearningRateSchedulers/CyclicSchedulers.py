from .LearningRateScheduler import LearningRateScheduler


class TriangularLearningRate(LearningRateScheduler):
    """ In Triangular Learing Rate, the learning rate will start at the min_lr and linearly increase
        to the max_lr and back down to min_lr in the timespan of 2 * stepsize
    """

    def __init__(self, min_lr, max_lr, stepsize):
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.stepsize = stepsize

        self.period = 2 * stepsize
        self.gradient = (max_lr - min_lr) / stepsize
        self.epoch_counter = 0

    def get(self):
        """ Note: It is very easy to make this method better"""
        mod_epoch = self.epoch_counter % self.period
        self.epoch_counter += 1

        if mod_epoch < self.stepsize:
            return self.min_lr + mod_epoch * self.gradient
        return self.max_lr - (mod_epoch - self.stepsize) * self.gradient


class WelchLearningRate(LearningRateScheduler):
    """ In Welch Learning Rate, the learning rate will start at the min_lr and increase
        to the max_lr and back down to min_lr in a parabolic fashion, within the timespan of
        2 * stepsize
    """

    def __init__(self, min_lr, max_lr, stepsize):
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.stepsize = stepsize

        self.amplitude = max_lr - min_lr
        self.period = 2 * stepsize
        self.epoch_counter = 0

    def get(self):
        mod_epoch = self.epoch_counter % self.period
        self.epoch_counter += 1

        welch_window = 1 - (mod_epoch / self.stepsize - 1) ** 2
        return self.min_lr + welch_window * self.amplitude
