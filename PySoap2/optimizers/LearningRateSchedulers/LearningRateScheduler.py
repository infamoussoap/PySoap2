import abc


class LearningRateScheduler(abc.ABC):
    @abc.abstractmethod
    def get(self):
        """ Returns the learning rate, governed by the scheduler """
        pass
