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
    # if its all, return all distribution errors
    if error_name == 'all':
        return filters
    else:
        # return error if exist
        return dict({error_name: filters[error_name]}) if error_name in filters else dict()


def dithering(img, error, zigzag):
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

        # reverse range using ::-1
        for x in range(eX, dX)[::-1 if reverse else 1]:
            # thresholding
            out[y, x] = 0 if img[y, x] < 128 else 255

            diff = img[y, x] - out[y, x]
            slice = img[y:y+eY+1, x-eX:x+eX+1]
            # accumulate errors
            slice += (error_curr * diff)

    # remove border added
    out = out[:dY, eX:dX]
    return out


def halftones(file, error, channel='gray', zigzag=False):
    if channel == 'bgr':
        inp_img = cv.imread(file, cv.IMREAD_COLOR)
        out_img = np.zeros(inp_img.shape)
        
        # halftones in each channel color [bgr]
        for i in [0, 1, 2]:
            out_img[:, :, i] = dithering(inp_img[:, :, i], error, zigzag)
    elif channel == 'hsv':
        inp_img = cv.imread(file, cv.IMREAD_COLOR)
        inp_img = cv.cvtColor(inp_img, cv.COLOR_BGR2HSV)

        out_img = np.copy(inp_img)
        # halftones in each channel color [v]
        for i in [2]:
            out_img[:, :, i] = dithering(inp_img[:, :, i], error, zigzag)

        out_img = cv.cvtColor(out_img, cv.COLOR_HSV2BGR)
    else:
        inp_img = cv.imread(file, cv.IMREAD_GRAYSCALE)
        out_img = dithering(inp_img, error, zigzag)

    return inp_img, out_img


def main():
    parser = argparse.ArgumentParser(description='Halftones algorithm.')
    parser.add_argument(
        'in_image', help='Name of the input image: images/baboon.png')
    parser.add_argument(
        'in_distr', help='Name of error distribution: (floyd, stevenson, burkes, sierra, stucki, jarvis, all).')
    parser.add_argument(
        'channel', help='Channel color of the image: (gbr, hsv, gray).')
    parser.add_argument(
        '--zigzag', help='Use zigzag alternating.', action='store_true')
    args = parser.parse_args()

    errors = get_errors(args.in_distr)
    for error in errors:
        _, out_img = halftones(args.in_image, errors[error],
                               args.channel, args.zigzag)

        out_name = '_{}_{}_{}.'.format(
            error, 'zigzag' if args.zigzag else 'line', args.channel
        )
        out_image = out_name.join(args.in_image.rsplit('.'))
        cv.imwrite(out_image, out_img)
        print('{} saved: {}'.format(args.in_image, error))


if __name__ == "__main__":
    main()

# Example to tun program
# python3 halftones.py images/baboon.png floyd bgr
