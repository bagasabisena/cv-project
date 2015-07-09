__author__ = 'bagas'

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
                        if (idx[0] < 0 or idx[1] < 0
                            or idx[0] >= np.size(im, axis=0)
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
