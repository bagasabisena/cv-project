import numpy as np

__author__ = 'bagas'


class Grid:

    def __init__(self, data, index):
        self.data = data
        self.index = index


def create_grid(im, w, h):

    height = np.size(im, 0)
    width = np.size(im, 1)

    grid_size = (height / h) * (width / w)
    grids = []

    for i in range(0, height, h):
        for j in range(0, width, w):
            data = im[i:i+h, j:j+w]
            index = (range(i, i+h), range)
            grids.append(Grid())