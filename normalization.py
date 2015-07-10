from __future__ import division
import numpy as np
from scipy.stats import multivariate_normal
from scipy.ndimage import convolve
from PIL import Image
import utils


def gamma_correction(im, gamma):
    return pow(im, gamma)

# calculate 5x5 gaussian kernel
def compute_kernel(sigma):
    gauss2d = multivariate_normal(mean=[0, 0], cov=[[pow(sigma, 2),0],
                                                   [0, pow(sigma, 2)]])

    kernel = np.empty([5,5])
    for x in range(-2,3,1):
        for y in range(-2,3,1):
            kernel[x+2, y+2] = gauss2d.pdf((x,y))

    return kernel

def dog(im, sigma1, sigma2):

    # calculate 5x5 gaussian kernel
    kernel1 = compute_kernel(sigma1)
    kernel2 = compute_kernel(sigma2)

    # convolution, pad with zero on the edge of image
    gauss1 = convolve(im, kernel1, mode='constant', cval=0)
    gauss2 = convolve(im, kernel2, mode='constant', cval=0)

    return gauss1-gauss2

# two step contrast equalization to compensate for extreme illumination
def contrast_equalization(im, alpha, tau):

    # first step
    denom = pow(np.absolute(im), alpha)
    denom = np.mean(denom)
    denom = pow(denom, 1/alpha)
    im = im / denom

    # second step
    taus = tau * np.ones(im.shape)
    denom = np.minimum(taus, np.absolute(im))
    denom = pow(denom, alpha)
    denom = np.mean(denom)
    denom = pow(denom, 1/alpha)
    im = im / denom

    # force value into tanh
    im = tau * np.tanh(im / tau)
    return im


# the tan triggs normalization code
def tan_triggs_norm(im, gamma, sigma1, sigma2, alpha, tau):
    im = gamma_correction(im, gamma)
    im = dog(im, sigma1, sigma2)
    im = contrast_equalization(im, alpha, tau)
    return im

if __name__ == '__main__':
    im = Image.open('../data/YALE/centered/subject01.normal.pgm')
    im = utils.to_numpy(im)
    im_norm = tan_triggs_norm(im, 0.2, 1, 4, 0.1, 10)

