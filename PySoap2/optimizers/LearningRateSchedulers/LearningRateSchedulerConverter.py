from . import LearningRateScheduler
from . import ConstantLearningRate


def convert_to_learning_rate_scheduler(learning_rate):
    if isinstance(learning_rate, LearningRateScheduler):
        return learning_rate
    elif isinstance(learning_rate, (float, int)):
        return ConstantLearningRate(learning_rate)
    else:
        raise ValueError(f'{type(learning_rate)} is not a valid type for the learning rate.'
                         ' Either set learning_rate to be a float/int or an instance of LearningRateScheduler.')
