import numpy as np
import cv2 as cv


def get_perspective_transform(pointsA, pointsB):
    if pointsA.shape != (4, 2) or pointsB.shape != (4, 2):
        raise ValueError("There must be four points")

    a = np.zeros((8, 8))
    b = np.zeros((8))

    # system of linear equations (8x8)
    for i in range(4):
        a[i][0] = a[i+4][3] = pointsA[i][0]
        a[i][1] = a[i+4][4] = pointsA[i][1]

        a[i][2] = 1
        a[i+4][5] = 1

        a[i][6] = -pointsA[i][0]*pointsB[i][0]
        a[i+4][6] = -pointsA[i][0]*pointsB[i][1]

        a[i][7] = -pointsA[i][1]*pointsB[i][0]
        a[i+4][7] = -pointsA[i][1]*pointsB[i][1]

        b[i] = pointsB[i][0]
        b[i+4] = pointsB[i][1]

    # solve linear system
    x = np.linalg.solve(a, b)
    x.resize((9,))
    x[8] = 1

    return x.reshape((3, 3))


def warp_perspective(img, M, dsize):
    # indices [i, j, 1]
    iY, iX = np.indices(dimensions=dsize)
    indexOutput = np.stack(
        (iX.ravel(), iY.ravel(), np.ones(iY.size))).astype(int)

    # inverse matrix
    IM = np.linalg.inv(M)
    indexInput = IM.dot(indexOutput)
    indexInput /= indexInput[2, :]

    # nearest neighbor interpolation
    indexInput = np.round(indexInput).astype(int)

    # set zero value
    indexInput = indexInput.transpose()
    index = np.where((indexInput[:, 0] < 0) | (indexInput[:, 0] >= img.shape[0]) |
                     (indexInput[:, 1] < 0) | (indexInput[:, 1] >= img.shape[1]))
    indexInput[index] = 0

    out = img[indexInput[:, 1], indexInput[:, 0]]
    return out.reshape(dsize)


def main():
    pt1 = np.array([[37, 51], [342, 42], [485, 467], [73, 380]])
    pt2 = np.array([[0, 0], [511, 0], [511, 511], [0, 511]])

    input_path = './images/baboon_perspectiva.png'
    input_image = cv.imread(input_path, cv.IMREAD_GRAYSCALE)

    M = get_perspective_transform(pt1, pt2)
    output_image = warp_perspective(input_image, M, (512, 512))

    cv.imshow("Original", input_image)
    cv.imshow("Perspective", output_image.astype('uint8'))

    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
