from __future__ import division
import decimal
import math
import numpy as np
from PIL import Image
import utils
import normalization

def circle_point(cx, cy, angle, radius):
    
    decimal.getcontext().prec = 6
    
    x = cx + radius * math.cos(angle)
    y = cy + radius * math.sin(angle)
    return (decimal.Decimal(x), decimal.Decimal(y))

def circle_point2(cx, cy, angle, radius):

    x = cx + radius * math.cos(angle)
    y = cy + radius * math.sin(angle)
    return (x, y)


def circular_points(cx, cy, radius, point):
    
    base_angle = 2*math.pi / point
    points = [circle_point(cx, cy, base_angle*i, radius) for i in range(point)]
    return points


def circular_points2(cx, cy, radius, point):

    base_angle = 2*math.pi / point
    points = [circle_point2(cx, cy, base_angle*i, radius) for i in range(point)]
    return points

def is_integer(x):
    return x % 1 == 0


def is_integer2(x):
    epsilon = 0.0000001
    if abs(x - round(x)) < epsilon:
        return True
    else:
        return False

# bilinear interpolation from
# http://stackoverflow.com/questions/8661537/how-to-perform-bilinear-interpolation-in-python
def bilinear_interpolation(x, y, points):
    '''Interpolate (x,y) from values associated with four points.

    The four points are a list of four triplets:  (x, y, value).
    The four points can be in any order.  They should form a rectangle.

        >>> bilinear_interpolation(12, 5.5,
        ...                        [(10, 4, 100),
        ...                         (20, 4, 200),
        ...                         (10, 6, 150),
        ...                         (20, 6, 300)])
        165.0

    '''
    # See formula at:  http://en.wikipedia.org/wiki/Bilinear_interpolation

    points = sorted(points)               # order points by x, then by y
    (x1, y1, q11), (_x1, y2, q12), (x2, _y1, q21), (_x2, _y2, q22) = points

    if x1 != _x1 or x2 != _x2 or y1 != _y1 or y2 != _y2:
        raise ValueError('points do not form a rectangle')
    if not x1 <= x <= x2 or not y1 <= y <= y2:
        raise ValueError('(x, y) not within the rectangle')

    return (q11 * (x2 - x) * (y2 - y) +
            q21 * (x - x1) * (y2 - y) +
            q12 * (x2 - x) * (y - y1) +
            q22 * (x - x1) * (y - y1)
           ) / ((x2 - x1) * (y2 - y1) + decimal.Decimal(0.0))

def bilinear_interpolation2(x, y, points):
    '''Interpolate (x,y) from values associated with four points.

    The four points are a list of four triplets:  (x, y, value).
    The four points can be in any order.  They should form a rectangle.

        >>> bilinear_interpolation2(12, 5.5,
        ...                        [(10, 4, 100),
        ...                         (20, 4, 200),
        ...                         (10, 6, 150),
        ...                         (20, 6, 300)])
        165.0

    '''
    # See formula at:  http://en.wikipedia.org/wiki/Bilinear_interpolation

    points = sorted(points)               # order points by x, then by y
    (x1, y1, q11), (_x1, y2, q12), (x2, _y1, q21), (_x2, _y2, q22) = points

    if x1 != _x1 or x2 != _x2 or y1 != _y1 or y2 != _y2:
        raise ValueError('points do not form a rectangle')
    if not x1 <= x <= x2 or not y1 <= y <= y2:
        raise ValueError('(x, y) not within the rectangle')

    return (q11 * (x2 - x) * (y2 - y) +
            q21 * (x - x1) * (y2 - y) +
            q12 * (x2 - x) * (y - y1) +
            q22 * (x - x1) * (y - y1)
           ) / ((x2 - x1) * (y2 - y1) + 0.0)


def integer_ceil(x):
    if is_integer(x):
        return decimal.Decimal(x+1)
    else:
        return decimal.Decimal(math.ceil(x))

def integer_ceil2(x):
    if is_integer(x):
        return x+1
    else:
        return math.ceil(x)

# return a list of pattern which are uniform pattern
def uniform_decimal():
    uniform = []
    for dec in range(0, 256):
        bitstring = '{0:08b}'.format(dec)
        count = 0
        for i in range(8-1):
            if bitstring[i] != bitstring[i+1]:
                count += 1
        if count <= 2:
            uniform.append(dec)
    return uniform


def grid_lbp(im, point, radius, face_area, block_size):

    min_height = face_area[0]
    max_height = face_area[1]
    min_width = face_area[2]
    max_width = face_area[3]
    block_height = block_size[0]
    block_width = block_size[1]
    block_total = ((max_height - min_width) / block_height) * ((max_width - min_width) / block_width)
    histogram_width = pow(2, point)

    block = 0

    # initialize the histograms
    hists = np.empty(shape=(block_total, histogram_width))

    for i in range(min_height, max_height, block_height):
        for j in range(min_width, max_width, block_width):

            rows = range(i, i+block_height)
            cols = range(j, j+block_width)

            lbp_array = []

            for k in rows:
                for l in cols:
                    pixel = im[k, l]
                    neighbor_idx = circular_points(k, l, radius, point)

                    neighbor_val = np.zeros(point)

                    for pos, idx in enumerate(neighbor_idx):
                        # check if point is outside the image
                        if idx[0] < 0 or idx[1] < 0 or idx[0] >= np.size(im, axis=0) or idx[1] >= np.size(im, axis=1):
                            neighbor_val[pos] = 255
                        # integer point, get exact pixel value
                        elif is_integer(idx[0]) and is_integer(idx[1]):
                            neighbor_val[pos] = im[idx]
                        # require interpolation
                        else:
                            x = idx[0]
                            y = idx[1]

                            # get point for interpolation
                            # if the point is integer, ceiling = point+1
                            point1 = (decimal.Decimal(math.floor(x)),
                                      decimal.Decimal(math.floor(y)))
                            point2 = (decimal.Decimal(math.floor(x)),
                                      integer_ceil(y))
                            point3 = (integer_ceil(x),
                                      decimal.Decimal(math.floor(y)))
                            point4 = (integer_ceil(x), integer_ceil(y))

                            n = [point1, point2, point3, point4]

                            # check if all interpolated point is
                            # outside the image
                            # if yes, return -1 for value
                            for idx in n:
                                if idx[0] < 0 or idx[1] < 0 or idx[0] >= np.size(im, axis=0) or idx[1] >= np.size(im, axis=1):
                                    neighbor_val[pos] = 255
                                    break

                            if neighbor_val[pos] == 255:
                                break

                            n = [(point1[0], point1[1], im[point1]),
                                 (point2[0], point2[1], im[point2]),
                                 (point3[0], point3[1], im[point3]),
                                 (point4[0], point4[1], im[point4])]

                            neighbor_val[pos] = bilinear_interpolation(x, y, n)

                    is_larger_than = neighbor_val > pixel
                    bitstring = ''.join([str(int(bit)) for bit in is_larger_than])
                    dec_value = int(bitstring, 2)

                    # populate the lbp array
                    lbp_array.append(dec_value)

            hist = np.histogram(lbp_array, bins=histogram_width,
                                range=(0, histogram_width))

            norm_hist = hist[0] / sum(hist[0])
            hists[block, :] = norm_hist
            block += 1

    return hists

def uniform_lbp(im, radius, face_area, block_size):

    # only for 8 points neighbor
    point = 8

    min_height = face_area[0]
    max_height = face_area[1]
    min_width = face_area[2]
    max_width = face_area[3]
    block_height = block_size[0]
    block_width = block_size[1]
    block_total = ((max_height - min_width) / block_height) * ((max_width - min_width) / block_width)
    block = 0

    # uniform label for 8 points is 58. 1 bins for non uniform
    histogram_width = 59
    uniform_label = uniform_decimal()
    # initialize the histograms
    hists = np.empty(shape=(block_total, histogram_width))

    for i in range(min_height, max_height, block_height):
        for j in range(min_width, max_width, block_width):

            rows = range(i, i+block_height)
            cols = range(j, j+block_width)

            hist = np.zeros(histogram_width)

            for k in rows:
                for l in cols:
                    pixel = im[k, l]
                    neighbor_idx = circular_points(k, l, radius, point)

                    neighbor_val = np.zeros(point)

                    for pos, idx in enumerate(neighbor_idx):
                        # check if point is outside the image
                        if idx[0] < 0 or idx[1] < 0 or idx[0] >= np.size(im, axis=0) or idx[1] >= np.size(im, axis=1):
                            neighbor_val[pos] = 255
                        # integer point, get exact pixel value
                        elif is_integer(idx[0]) and is_integer(idx[1]):
                            neighbor_val[pos] = im[idx]
                        # require interpolation
                        else:
                            x = decimal.Decimal(idx[0])
                            y = decimal.Decimal(idx[1])

                            # get point for interpolation
                            # if the point is integer, ceiling = point+1
                            point1 = (decimal.Decimal(math.floor(x)),
                                      decimal.Decimal(math.floor(y)))
                            point2 = (decimal.Decimal(math.floor(x)),
                                      integer_ceil(y))
                            point3 = (integer_ceil(x),
                                      decimal.Decimal(math.floor(y)))
                            point4 = (integer_ceil(x), integer_ceil(y))

                            n = [point1, point2, point3, point4]

                            # check if all interpolated point is
                            # outside the image
                            # if yes, return -1 for value
                            for idx in n:
                                if idx[0] < 0 or idx[1] < 0 or idx[0] >= np.size(im, axis=0) or idx[1] >= np.size(im, axis=1):
                                    neighbor_val[pos] = 255
                                    break

                            if neighbor_val[pos] == 255:
                                break

                            n = [(point1[0], point1[1], decimal.Decimal(im[point1])),
                                 (point2[0], point2[1], decimal.Decimal(im[point2])),
                                 (point3[0], point3[1], decimal.Decimal(im[point3])),
                                 (point4[0], point4[1], decimal.Decimal(im[point4]))]

                            neighbor_val[pos] = bilinear_interpolation(x, y, n)

                    is_larger_than = neighbor_val > pixel
                    bitstring = ''.join([str(int(bit)) for bit in is_larger_than])
                    dec_value = int(bitstring, 2)

                    # populate the histogram
                    # index function should return the index of the bin
                    # if the decimal is uniform
                    # otherwise it returns TypeError
                    try:
                        bins = uniform_label.index(dec_value)
                        hist[bins] += 1
                    except ValueError:
                        # populate the 59 bin for non uniform
                        hist[58] += 1

            norm_hist = hist / sum(hist)
            hists[block, :] = norm_hist
            block += 1

    return hists

def uniform_lbp2(im, radius, face_area, block_size):

    # only for 8 points neighbor
    point = 8

    min_height = face_area[0]
    max_height = face_area[1]
    min_width = face_area[2]
    max_width = face_area[3]
    block_height = block_size[0]
    block_width = block_size[1]
    block_total = ((max_height - min_height) / block_height) * ((max_width - min_width) / block_width)
    block = 0

    im_height = np.size(im, axis=0)
    im_width = np.size(im, axis=1)

    # uniform label for 8 points is 58. 1-58th bins for uniform
    # 59th bin for non uniform
    histogram_width = 59
    uniform_label = uniform_decimal()
    # initialize the histograms
    hists = np.empty(shape=(block_total, histogram_width))

    for i in range(min_height, max_height, block_height):
        for j in range(min_width, max_width, block_width):

            rows = range(i, i+block_height)
            cols = range(j, j+block_width)

            hist = np.zeros(histogram_width)

            for k in rows:
                for l in cols:
                    pixel = im[k, l]
                    neighbor_idx = circular_points2(k, l, radius, point)

                    neighbor_val = np.zeros(point)

                    for pos, idx in enumerate(neighbor_idx):
                        # check if point is outside the image
                        if idx[0] < 0 or idx[1] < 0 or idx[0] >= im_height or idx[1] >= im_width:
                            neighbor_val[pos] = 255
                        # integer point, get exact pixel value
                        elif is_integer2(idx[0]) and is_integer2(idx[1]):
                            idx = (round(idx[0]), round(idx[1]))
                            neighbor_val[pos] = im[idx]
                        # require interpolation
                        else:
                            x = idx[0]
                            y = idx[1]

                            # get point for interpolation
                            # if the point is integer, ceiling = point+1
                            point1 = (math.floor(x),
                                      math.floor(y))
                            point2 = (math.floor(x),
                                      integer_ceil2(y))
                            point3 = (integer_ceil2(x),
                                      math.floor(y))
                            point4 = (integer_ceil2(x), integer_ceil2(y))

                            n = [point1, point2, point3, point4]

                            # check if all interpolated point is
                            # outside the image
                            # if yes, return -1 for value
                            for idx in n:
                                if idx[0] < 0 or idx[1] < 0 or idx[0] >= im_height or idx[1] >= im_width:
                                    neighbor_val[pos] = 255
                                    break

                            if neighbor_val[pos] == 255:
                                break

                            n = [(point1[0], point1[1], im[point1]),
                                 (point2[0], point2[1], im[point2]),
                                 (point3[0], point3[1], im[point3]),
                                 (point4[0], point4[1], im[point4])]

                            neighbor_val[pos] = bilinear_interpolation2(x, y, n)

                    is_larger_than = neighbor_val > pixel
                    bitstring = ''.join([str(int(bit)) for bit in is_larger_than])
                    dec_value = int(bitstring, 2)

                    # populate the histogram
                    # index function should return the index of the bin
                    # if the decimal is uniform
                    # otherwise it returns TypeError
                    try:
                        bins = uniform_label.index(dec_value)
                        hist[bins] += 1
                    except ValueError:
                        # populate the 59 bin for non uniform
                        hist[58] += 1

            norm_hist = hist / sum(hist)
            hists[block, :] = norm_hist
            block += 1

    return hists


def circular_lbp(im, point, radius):
    
    row = np.size(im, 0)
    col = np.size(im, 1)
    
    lbp_im = np.array(im, copy=True)
    
    for i in range(0, row):
        for j in range(0, col):
            pixel = im[i, j]
            neighbor_idx = circular_points(i, j, radius, point)
            
            neighbor_val = np.zeros(point)
                
            for i,idx in enumerate(neighbor_idx):
                # check if point is outside the image
                if (idx[0] < 0 or idx[1] < 0
                    or idx[0] >= np.size(im, axis=0)
                    or idx[1] >= np.size(im, axis=1)):

                    neighbor_val[i] = -1
                # integer point, get exact pixel value
                elif is_integer(idx[0]) and is_integer(idx[1]):
                    neighbor_val[i] = im[idx]
                # require interpolation
                else:
                    x = idx[0]
                    y = idx[1]
                    
                    # get point for interpolation
                    # if the point is integer, ceiling = point+1
                    point1 = (decimal.Decimal(math.floor(x)),
                              decimal.Decimal(math.floor(y)))
                    point2 = (decimal.Decimal(math.floor(x)), integer_ceil(y))
                    point3 = (integer_ceil(x), decimal.Decimal(math.floor(y)))
                    point4 = (integer_ceil(x), integer_ceil(y))
                    
                    n = [point1, point2, point3, point4]
                    
                    # check if all interpolated point is outside the image
                    # if yes, return -1 for value
                    for idx in n:
                        if (idx[0] < 0 or idx[1] < 0 or idx[0] >= np.size(im, axis=0)
                            or idx[1] >= np.size(im, axis=1)):
                            neighbor_val[i] = -1
                            break
                    
                    if neighbor_val[i] == -1:
                        break
                    
                    n = [(point1[0], point1[1], im[point1]),
                         (point2[0], point2[1], im[point2]),
                         (point3[0], point3[1], im[point3]),
                         (point4[0], point4[1], im[point4])] 
                         
                    neighbor_val[i] = bilinear_interpolation(x, y, n)
            
            is_larger_than = neighbor_val > pixel
            bitstring = ''.join([str(int(bit)) for bit in is_larger_than])
            dec_value = int(bitstring, 2)
            
            # populate the lbp array
            lbp_im[i, j] = dec_value
            
    return lbp_im
    
    
if __name__ == '__main__':
    im = Image.open('../data/YALE/centered/subject01.centerlight.pgm')
    im = utils.to_numpy(im)
    face_area = (15, 215, 15, 175)
    block_size = (40, 40)
    radius = 2
    im = normalization.tan_triggs_norm(im, 0.2, 1, 4, 0.1, 10)
    lbp_uni = uniform_lbp2(im, radius, face_area, block_size)
