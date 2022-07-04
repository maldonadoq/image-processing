import numpy as np
import cv2 as cv
import argparse
import time


# nearest neighbor interpolation
def inter_nearest(img, indices):
    # round method
    indices = np.round(indices).astype(int)

    # set zero value
    indices = indices.transpose()
    index = np.where((indices[:, 0] < 0) | (indices[:, 0] >= img.shape[1]) |
                     (indices[:, 1] < 0) | (indices[:, 1] >= img.shape[0]))
    indices[index] = 0
    return img[indices[:, 1], indices[:, 0]]


# bilinear interpolation
def inter_bilinear(img, indices):
    x = indices[0, :]
    y = indices[1, :]

    # floor index x, y
    # (x0, y0) - (x1, y1)
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    x1 = x0 + 1
    y1 = y0 + 1

    # limit indices between img.shape
    x0 = np.clip(x0, 0, img.shape[1]-1)
    y0 = np.clip(y0, 0, img.shape[0]-1)
    x1 = np.clip(x1, 0, img.shape[1]-1)
    y1 = np.clip(y1, 0, img.shape[0]-1)

    # four element from bilinear equation
    A = (x1-x) * (y1-y) * img[y0, x0]
    B = (x1-x) * (y-y0) * img[y1, x0]
    C = (x-x0) * (y1-y) * img[y0, x1]
    D = (x-x0) * (y-y0) * img[y1, x1]

    return A + B + C + D


# spline function
def u(s, a=-0.5):
    st = abs(s)
    if st <= 1:
        return (a+2)*(st**3) - (a+3)*(st**2) + 1
    elif st < 2:
        return a*(st**3) - (5*a)*(st**2) + (8*a)*st - 4*a
    return 0


# bicubic interpolation
def inter_bicubic(img, indices):
    # filter valid indices
    indices = indices.transpose()
    index = np.where((indices[:, 0] >= 1) & (indices[:, 0] < img.shape[1]-2) &
                     (indices[:, 1] >= 1) & (indices[:, 1] < img.shape[0]-2))

    # new indices with valid indices
    newIndices = indices[index]
    newValues = np.empty(newIndices.shape[0])

    for i in range(newIndices.shape[0]):
        # floor values
        point = newIndices[i]
        pointF = np.floor(point)

        # delta values
        pointD = point - pointF

        # four values from index x
        x1 = pointD[0] + 1
        x2 = pointD[0]
        x3 = -pointD[0] + 1
        x4 = -pointD[0] + 2

        # four values from index y
        y1 = pointD[1] + 1
        y2 = pointD[1]
        y3 = -pointD[1] + 1
        y4 = -pointD[1] + 2

        A = np.array([u(x1), u(x2), u(x3), u(x4)])
        B = np.array([
            [img[int(point[1]-y1), int(point[0]-x1)], img[int(point[1]-y2), int(point[0]-x1)],
             img[int(point[1]+y3), int(point[0]-x1)], img[int(point[1]+y4), int(point[0]-x1)]],
            [img[int(point[1]-y1), int(point[0]-x2)], img[int(point[1]-y2), int(point[0]-x2)],
             img[int(point[1]+y3), int(point[0]-x2)], img[int(point[1]+y4), int(point[0]-x2)]],
            [img[int(point[1]-y1), int(point[0]+x3)], img[int(point[1]-y2), int(point[0]+x3)],
             img[int(point[1]+y3), int(point[0]+x3)], img[int(point[1]+y4), int(point[0]+x3)]],
            [img[int(point[1]-y1), int(point[0]+x4)], img[int(point[1]-y2), int(point[0]+x4)],
             img[int(point[1]+y3), int(point[0]+x4)], img[int(point[1]+y4), int(point[0]+x4)]]
        ])
        C = np.array([u(y1), u(y2), u(y3), u(y4)])
        newValues[i] = np.dot(np.dot(A, B), C)

    # set new values
    out = np.zeros(indices.shape[0])
    out[index] = newValues
    return out


# lagrange expansion
def expansion(img, pointF, pointD, n):
    a = (-pointD[0] * (pointD[0]-1) * (pointD[0]-2) *
         img[int(pointF[1]-1), int(pointF[0]+n-2)]) / 6
    b = ((pointD[0]+1) * (pointD[0]-1) * (pointD[0]-2)
         * img[int(pointF[1]), int(pointF[0]+n-2)]) / 2
    c = (-pointD[0] * (pointD[0]+1) * (pointD[0]-2) *
         img[int(pointF[1]+1), int(pointF[0]+n-2)]) / 2
    d = (pointD[0] * (pointD[0]+1) * (pointD[0]-1) *
         img[int(pointF[1]+2), int(pointF[0]+n-2)]) / 6

    return a + b + c + d


# lagrange polynomial interpolation
def inter_lagrange(img, indices):
    # filter valid indices
    indices = indices.transpose()
    index = np.where((indices[:, 0] >= 1) & (indices[:, 0] < img.shape[1]-2) &
                     (indices[:, 1] >= 1) & (indices[:, 1] < img.shape[0]-2))

    # new indices with valid indices
    newIndices = indices[index]
    newValues = np.empty(newIndices.shape[0])

    for i in range(newIndices.shape[0]):
        # floor values
        point = newIndices[i]
        pointF = np.floor(point)
        pointD = point - pointF

        a = (-pointD[1] * (pointD[1]-1) * (pointD[1]-2)
             * expansion(img, pointF, pointD, 1)) / 6
        b = ((pointD[1]+1) * (pointD[1]-1) * (pointD[1]-2)
             * expansion(img, pointF, pointD, 2)) / 2
        c = (-pointD[1] * (pointD[1]+1) * (pointD[1]-2)
             * expansion(img, pointF, pointD, 3)) / 2
        d = (pointD[1] * (pointD[1]+1) * (pointD[1]-1)
             * expansion(img, pointF, pointD, 4)) / 6

        newValues[i] = a + b + c + d

    # set new values
    out = np.zeros(indices.shape[0])
    out[index] = newValues
    return out


# apply transform
def transform(img, M, dsize, inter):
    # indices [i, j, 1]
    iY, iX = np.indices(dimensions=dsize)
    indexOutput = np.stack(
        (iX.ravel(), iY.ravel(), np.ones(iY.size))).astype(int)

    # inverse matrix
    IM = np.linalg.inv(M)
    indexInput = IM.dot(indexOutput)
    indexInput /= indexInput[2, :]

    # interpolation type
    if inter == 'nearest':
        out = inter_nearest(img, indexInput)
    elif inter == 'bilinear':
        out = inter_bilinear(img, indexInput)
    elif inter == 'bicubic':
        out = inter_bicubic(img, indexInput)
    elif inter == 'lagrange':
        out = inter_lagrange(img, indexInput)
    else:
        return np.zeros(dsize)

    return out.reshape(dsize)


def get_dim(dim, shape):
    if dim:
        dims = dim.split(',')
        if len(dims) == 2:
            return (int(dims[0]), int(dims[1]))
    return shape


def get_values(value):
    values = value.split(',')
    if len(values) == 2:
        try:
            return [float(values[0]), float(values[1])]
        except ValueError:
            raise
    raise TypeError('{} is not a tuple of 2 elements (x,y)'.format(value))


def main():
    parser = argparse.ArgumentParser(description='Interpolation techniques.')
    parser.add_argument(
        '-a', '--angle', dest='angle', type=float,  help='Rotation angle (degrees)')
    parser.add_argument(
        '-s', '--scale', dest='scale', type=str, help='Scale factor')
    parser.add_argument(
        '-t', '--translation', dest='translation', type=str, help='Translation values')
    parser.add_argument(
        '-r', '--reflection', dest='reflection', type=str, help='Reflection axis (1,-1)')
    parser.add_argument(
        '-k', '--shear', dest='shear', type=str, help='Shear values')
    parser.add_argument(
        '-d', '--dimension', dest='dim', type=str, help='Output image dimension in pixels: (N,M)')
    parser.add_argument(
        '-m', '--method', dest='method', type=str, default='nearest', help='Interpolation method used: (nearest, bilinear, bicubic, lagrange)')
    parser.add_argument(
        '-i', '--image', dest='image', type=str, required=True, help='Input image in PNG format')
    parser.add_argument(
        '-o', '--output', dest='output', type=str, required=True, help='Output image in PNG format')

    args = parser.parse_args()

    input_path = args.image
    input_image = cv.imread(input_path, cv.IMREAD_GRAYSCALE)

    dimension = get_dim(args.dim, input_image.shape)
    output_image = np.zeros(dimension, dtype='uint8')

    transformation = 'scale'
    T = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])

    if args.angle:
        transformation = 'rotation'
        values = [-np.deg2rad(args.angle)]
        T = np.array([
            [np.cos(values[0]), -np.sin(values[0]), 0],
            [np.sin(values[0]), np.cos(values[0]), 0],
            [0, 0, 1]
        ])
    elif args.scale:
        transformation = 'scale'
        values = get_values(args.scale)
        T = np.array([
            [values[0], 0, 0],
            [0, values[1], 0],
            [0, 0, 1]
        ])
    elif args.translation:
        transformation = 'translation'
        values = get_values(args.translation)
        T = np.array([
            [1, 0, values[0]],
            [0, 1, values[1]],
            [0, 0, 1]
        ])
    elif args.reflection:
        transformation = 'reflection'
        values = get_values(args.reflection)
        T = np.array([
            [values[0], 0, 0],
            [0, values[1], 0],
            [0, 0, 1]
        ])
    elif args.shear:
        transformation = 'shear'
        values = get_values(args.shear)
        T = np.array([
            [1, values[0], 0],
            [values[1], 1, 0],
            [0, 0, 1]
        ])

    start = time.time()
    output_image = transform(input_image, T, dimension, args.method)
    end = time.time()
    output_path = args.output

    cv.imwrite(output_path, output_image)
    print('transformation: {} {}, interpolation: {}, time: {:.4} s.'.format(
        transformation, values, args.method, end - start))


if __name__ == "__main__":
    main()

# Example to tun program
# python3 interpolation.py -i ./images/baboon.png -o ./images/baboon_bilinear.png -s=0.75,0.75 -m bilinear
