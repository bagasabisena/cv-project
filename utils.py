__author__ = 'bagas'

from PIL import Image
import numpy as np

def to_numpy(image):

    pil_im = image.convert('L')
    return np.array(pil_im, dtype='uint8')
