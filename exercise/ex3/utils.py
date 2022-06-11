import cv2 as cv
import numpy as np


# save video summarized
def save_metrics(file, output, peaks):
    cap = cv.VideoCapture(file)

    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv.VideoWriter_fourcc(*'XVID')
    out = cv.VideoWriter(output, fourcc, 20.0, (width,  height))

    count = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            if count < len(peaks) and peaks[count]:
                out.write(frame)
        else:
            cap.release()
            out.release()
            break
        count += 1


# get all peaks
def threshold(metrics):
    """ ordered = np.sort(metrics)
    init = int(metrics.shape[0] * 0.8)
    return metrics, metrics > ordered[init] """

    # locals maxima vectorized
    peaks = np.zeros(metrics.shape, dtype=bool)
    peaks[1:-1] = np.diff(np.sign(np.diff(metrics))) < 0

    # middle locals maxima
    index = np.sort(metrics[peaks.nonzero()[0]])
    middle = metrics > index[index.shape[0] // 2]
    return peaks & middle


# process gray video in shots
def get_metrics_gray(file, shots_size, difference, params, dtype='uint8'):
    cap = cv.VideoCapture(file)

    cols = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    rows = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    shots = np.empty((shots_size + 1, rows, cols), dtype)
    metrics = []

    ret, frame = cap.read()
    first_frame = frame[:, :, 0]
    count = 1

    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            gray_frame = frame[:, :, 0]
            if count > shots_size:
                shots[0] = first_frame
                first_frame = shots[-1].copy()

                # process shots
                metrics += difference(shots, params)

                shots[1] = gray_frame
                count = 2
            else:
                shots[count] = gray_frame
                count += 1
        else:
            # process shots[:count,:,:]
            shots[0] = first_frame
            metrics += difference(shots[:count, :, :], params)
            cap.release()
            break

    metrics = np.array(metrics)
    return metrics, threshold(metrics)


# process color video in shots
def get_metrics_color(file, shots_size, difference, params, dtype='uint8'):
    cap = cv.VideoCapture(file)

    cols = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    rows = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    shots = np.empty((shots_size + 1, rows, cols), dtype)
    metrics = []

    ret, frame = cap.read()
    first_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    count = 1

    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            if count > shots_size:
                shots[0] = first_frame
                first_frame = shots[-1].copy()

                # process shots
                metrics += difference(shots, params)

                shots[1] = gray_frame
                count = 2
            else:
                shots[count] = gray_frame
                count += 1
        else:
            # process shots[:count,:,:]
            shots[0] = first_frame
            metrics += difference(shots[:count, :, :], params)
            cap.release()
            break

    metrics = np.array(metrics)
    return metrics, threshold(metrics)


# frame 2D into array of blocks
def reshape(image, rows, cols, size):
    rows, cols = image.shape

    tiled_array = image.reshape(rows // size, size, cols // size, size)
    tiled_array = tiled_array.swapaxes(1, 2).reshape(-1, size, size)
    return tiled_array


# normaliza frame 2D
def normalize(img_in):
    img_out = np.zeros(img_in.shape, dtype="uint8")

    tmin = img_in.min()
    tmax = img_in.max()

    tsize = abs(tmax - tmin)
    img_out = ((img_in - tmin) * 255) // tsize
    return img_out


# sobel filter X
def sobelX(img_in):
    res_h = img_in[:, 2:] - img_in[:, :-2]
    res_v = res_h[:-2] + res_h[2:] + 2*res_h[1:-1]
    return np.abs(res_v)


# sobel filter Y
def sobelY(img_in):
    img = img_in.transpose()
    res_h = img[:, 2:] - img[:, :-2]
    res_v = res_h[:-2] + res_h[2:] + 2*res_h[1:-1]
    return np.abs(res_v.transpose())


# sobel filter in frame 2D
def sobel(img):
    img = img.astype('int16')
    out = np.zeros(img.shape, int)
    x = sobelX(img)
    y = sobelY(img)

    out[1:-1, 1:-1] = x + y

    return out
