import numpy as np
import random

import scipy


def is_iterable(index):
    try:
        _ = (i for i in index)
    except TypeError:
        return False
    else:
        return True


def rotate(x):
    angle = np.random.uniform(low=-10, high=10)
    return scipy.ndimage.rotate(x, angle)


def gaussian_blur(x):
    sigma = np.random.uniform(low=0, high=0.7)
    return scipy.ndimage.gaussian_filter(x, sigma)


def identity(x):
    return x


class ImageAugmentationGenerator:
    """ Class that generates random augmentations of the dataset. It should support any indexing that numpy supports.

        Current Augmentations:
            Flip left-right, Flip up-down, Rotation of +-10 degrees, Gaussian Blur and Identity

        Note that augmentation is generated when __getitem__ is accessed.
    """
    def __init__(self, images):
        self.images = images
        self.image_augmentations = [np.fliplr, np.flipud, rotate, gaussian_blur, identity]

    def __getitem__(self, index):
        if is_iterable(index):
            return np.array([self.random_augmentation_function(self.images[i])
                             for i in index])

        elif isinstance(index, slice):
            return np.array([self.random_augmentation_function(image)
                             for image in self.images[index]])

        # Assume it is just an integer
        return self.random_augmentation_function(self.images[index])

    def __setitem__(self, *args, **kwargs):
        self.images.__setitem__(*args, **kwargs)

    def __delitem__(self, *args, **kwargs):
        self.images.__delitem__(*args, **kwargs)

    @property
    def random_augmentation_function(self):
        return random.choice(self.image_augmentations)