__author__ = 'bagas'

import numpy as np


# convert pgm image from YALE dataset to numpy array
def to_numpy(image):

    pil_im = image.convert('L')
    return np.array(pil_im, dtype='uint8')

