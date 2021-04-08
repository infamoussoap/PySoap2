import numpy as np
import random

import scipy

from .FancyPCA import fancy_pca


def is_iterable(index):
    try:
        _ = (i for i in index)
    except TypeError:
        return False
    else:
        return True


def rotate(x):
    height, width = x.shape[:2]

    angle = np.random.uniform(low=-10, high=10)
    return scipy.ndimage.rotate(x, angle, reshape=False)


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
        self.image_augmentations = [np.fliplr, np.flipud, gaussian_blur, identity]
        self.image_augmentations += [fancy_pca, fancy_pca, fancy_pca]

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

    def __len__(self):
        return len(self.images)

    def augment_images(self, images):
        return np.array([self.random_augmentation_function(image) for image in images])

    def __call__(self, images):
        return self.augment_images(images)
