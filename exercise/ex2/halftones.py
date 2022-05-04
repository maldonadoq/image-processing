from distutils import errors
import numpy as np
import cv2 as cv
import argparse

filters = {
    'floyd': 1/16 * np.array([
        [0, 0, 7],
        [3, 5, 1]
    ]),
    'stevenson': 1/200 * np.array([
        [0, 0, 0, 0, 0, 32, 0],
        [12, 0, 26, 0, 30, 0, 16],
        [0, 12, 0, 26, 0, 12, 0],
        [5, 0, 12, 0, 12, 0, 5]]),
    'burkes': 1/32 * np.array([
        [0, 0, 0, 8, 4],
        [2, 4, 8, 4, 2]
    ]),
    'sierra': 1/32 * np.array([
        [0, 0, 0, 5, 3],
        [2, 4, 5, 4, 2],
        [0, 2, 3, 2, 0]
    ]),
    'stucki': 1/42 * np.array([
        [0, 0, 0, 8, 4],
        [2, 4, 8, 4, 2],
        [1, 2, 4, 2, 1]
    ]),
    'jarvis': 1/48 * np.array([
        [0, 0, 0, 7, 5],
        [3, 5, 7, 5, 3],
        [1, 3, 5, 3, 1]
    ])
}


def get_errors(error_name):
    if error_name == 'all':
        return filters
    else:
        return dict({error_name: filters[error_name]}) if error_name in filters else dict()


def stipple(img, error, zigzag):
    # error distribution size
    eX = error.shape[1] // 2
    eY = error.shape[0] - 1

    # add border and change array type to float
    # f(x,y) image
    img = cv.copyMakeBorder(img,
                            top=0, bottom=eY,
                            left=eX, right=eX,
                            borderType=cv.BORDER_CONSTANT
                            )
    img = img.astype(float, copy=False)

    # size for loop
    dX = img.shape[1] - eX
    dY = img.shape[0] - eY

    # g(x,y) image
    out = np.zeros(img.shape)

    # reverse error
    error_rev = np.fliplr(error)

    for y in range(dY):
        # reverse if is odd
        reverse = (y % 2 == 1) if zigzag else False
        error_curr = error_rev if reverse else error

        for x in range(eX, dX)[::-1 if reverse else 1]:
            out[y, x] = 0 if img[y, x] < 128 else 255
            #img[y, x] = max(0, img[y, x])
            #out[y, x] = 255 * np.floor(img[y, x]/128)

            diff = img[y, x] - out[y, x]
            slice = img[y:y+eY+1, x-eX:x+eX+1]
            slice += (error_curr * diff)

    out = out[:dY, eX:dX]
    return out


def halftones(file, error, color=False, zigzag=False):
    if color:
        inp_img = cv.imread(file, cv.IMREAD_COLOR)
        out_img = np.zeros(inp_img.shape)
        # halftones in each field color
        for i in range(inp_img.shape[2]):
            out_img[:, :, i] = stipple(inp_img[:, :, i], error, zigzag)
    else:
        inp_img = cv.imread(file, cv.IMREAD_GRAYSCALE)
        out_img = stipple(inp_img, error, zigzag)

    return inp_img, out_img


def main():
    parser = argparse.ArgumentParser(description='Halftones algorithm.')
    parser.add_argument(
        'in_image', help='Name of the input image: images/baboon.png')
    parser.add_argument(
        'in_distr', help='Name of error distribution: (floyd, stevenson, burkes, sierra, stucki, jarvis, all).')
    parser.add_argument(
        '--zigzag', help='Use zigzag alternating.', action='store_true')
    parser.add_argument(
        '--color', help='Use color images or grayscale.', action='store_true')
    args = parser.parse_args()

    errors = get_errors(args.in_distr)
    for error in errors:
        _, out_img = halftones(args.in_image, errors[error],
                               args.color, args.zigzag)

        out_name = '_{}_{}_{}.'.format(error,
                                       'zigzag' if args.zigzag else 'line',
                                       'color' if args.color else 'gray'
                                       )
        out_image = out_name.join(args.in_image.rsplit('.'))
        cv.imwrite(out_image, out_img)
        print('{} saved: {}'.format(args.in_image, error))


if __name__ == "__main__":
    main()

# Example to tun program
# python3 halftones.py images/baboon.png all
