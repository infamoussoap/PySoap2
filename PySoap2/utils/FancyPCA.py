# I wrote none of this code. So big thanks to pixelatedbrian on github
# https://github.com/pixelatedbrian/fortnight-furniture/blob/master/src/fancy_pca.py

import numpy as np


def fancy_pca(img, alpha_std=0.03*255):
    """
    CREDIT:
    https://github.com/pixelatedbrian/fortnight-furniture/blob/master/src/fancy_pca.py

    INPUTS:
    img:  numpy array with (h, w, rgb) shape, as floats between 0-1)
    alpha_std:  how much to perturb/scale the eigen vecs and vals
                the paper used std=0.1
    RETURNS:
    numpy image-like array as float range(0, 1)
    NOTE: Depending on what is originating the image data and what is receiving
    the image data returning the values in the expected form is very important
    in having this work correctly. If you receive the image values as UINT 0-255
    then it's probably best to return in the same format. (As this
    implementation does). If the image comes in as float values ranging from
    0.0 to 1.0 then this function should be modified to return the same.
    Otherwise this can lead to very frustrating and difficult to troubleshoot
    problems in the image processing pipeline.
    This is 'Fancy PCA' from:
    # http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
    #######################
    #### FROM THE PAPER ###
    #######################
    "The second form of data augmentation consists of altering the intensities
    of the RGB channels in training images. Specifically, we perform PCA on the
    set of RGB pixel values throughout the ImageNet training set. To each
    training image, we add multiples of the found principal components, with
    magnitudes proportional to the corresponding eigenvalues times a random
    variable drawn from a Gaussian with mean zero and standard deviation 0.1.
    Therefore to each RGB image pixel Ixy = [I_R_xy, I_G_xy, I_B_xy].T
    we add the following quantity:
    [p1, p2, p3][α1λ1, α2λ2, α3λ3].T
    Where pi and λi are ith eigenvector and eigenvalue of the 3 × 3 covariance
    matrix of RGB pixel values, respectively, and αi is the aforementioned
    random variable. Each αi is drawn only once for all the pixels of a
    particular training image until that image is used for training again, at
    which point it is re-drawn. This scheme approximately captures an important
    property of natural images, namely, that object identity is invariant to
    change."
    ### END ###############
    Other useful resources for getting this working:
    # https://groups.google.com/forum/#!topic/lasagne-users/meCDNeA9Ud4
    # https://gist.github.com/akemisetti/ecf156af292cd2a0e4eb330757f415d2

    CREDIT:
    https://github.com/pixelatedbrian/fortnight-furniture/blob/master/src/fancy_pca.py
    """

    orig_img = img.astype(float).copy()

    # img = img / 255.0  # rescale to 0 to 1 range

    # flatten image to columns of RGB
    img_rs = img.reshape(-1, 3)

    # center mean
    img_centered = img_rs - np.mean(img_rs, axis=0)

    # paper says 3x3 covariance matrix
    img_cov = np.cov(img_centered, rowvar=False)

    # eigen values and eigen vectors
    eig_vals, eig_vecs = np.linalg.eigh(img_cov)

    # sort values and vector
    sort_perm = eig_vals[::-1].argsort()
    eig_vals[::-1].sort()
    eig_vecs = eig_vecs[:, sort_perm]

    # get [p1, p2, p3]
    m1 = np.column_stack(eig_vecs)

    # get 3x1 matrix of eigen values multiplied by random variable draw from normal
    # distribution with mean of 0 and standard deviation of 0.1
    m2 = np.zeros((3, 1))
    # according to the paper alpha should only be draw once per augmentation (not once per channel)
    alpha = np.random.normal(0, alpha_std)

    # broad cast to speed things up
    m2[:, 0] = alpha * eig_vals[:]

    # this is the vector that we're going to add to each pixel in a moment
    add_vect = m1@m2

    for idx in range(3):  # RGB
        orig_img[..., idx] += add_vect[idx]

    orig_img = np.clip(orig_img, 0.0, 255.0)

    max_ = np.max(orig_img)
    if max_ > 1e-5:  # Prevent divide by 0 errors
        orig_img = orig_img/np.max(orig_img)

    return orig_img
