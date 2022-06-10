
from cProfile import label
from cv2 import threshold
from matplotlib import pyplot as plt

import numpy as np
import argparse
import utils


def diff_pixels(frames, params):
    difference = np.abs(frames[1:] - frames[:-1])
    metrics = list(np.sum(difference > params[0], axis=(1, 2)))

    return metrics


def diff_blocks(frames, params):
    rows, cols = frames.shape[1], frames.shape[2]

    trows = rows // params[0]
    tcols = cols // params[0]

    frames = frames[:, :trows * params[0], :tcols * params[0]]
    difference = np.abs(frames[1:] - frames[:-1])
    difference = difference * difference

    metrics = []
    for diff in difference:
        soma = np.sum(utils.reshape(
            diff, trows, tcols, params[0]), axis=(1, 2))
        metrics.append((np.sqrt(soma) > params[1]).sum())
    return metrics


def diff_histogram(frames, params):
    total = frames.shape[0]
    histograms = np.empty((total, 256))

    for i in range(total):
        hist, _ = np.histogram(frames[i], bins=256, range=(0, 255))
        histograms[i] = hist

    difference = np.abs(histograms[1:] - histograms[:-1])
    means = np.mean(difference, axis=1)
    stds = np.std(difference, axis=1)
    metrics = means + stds*3

    return list(metrics)


def diff_edges(frames, params):
    metrics = []
    for frame in frames:
        mask = utils.normalize(utils.sobel(frame))
        metrics.append((mask > params[0]).sum())
    return metrics


def process(filename, shot_size, diff):
    metrics = None
    if diff == 'pixels':
        metrics = utils.get_metrics(
            filename, shot_size, diff_pixels, [128], 'int16')
    elif diff == 'blocks':
        metrics = utils.get_metrics(
            filename, shot_size, diff_blocks, [8, 128], 'int32')
    elif diff == 'histograms':
        metrics = utils.get_metrics(
            filename, shot_size, diff_histogram, [])
    elif diff == 'edges':
        metrics = utils.get_metrics(
            filename, shot_size, diff_edges, [128])
    else:
        raise NameError("'{}' > not defined".format(diff))

    return metrics


def save_metrics(metrics, peaks, output):
    X = np.array(range(len(metrics)))

    plt.figure(figsize=(25, 10))
    index = peaks.nonzero()[0]
    print('{} summarized'.format((index.shape[0] / metrics.shape[0]) * 100))
    plt.plot(X[index], metrics[index], "o", label="peaks")
    plt.plot(X, metrics, label="metrics")
    plt.legend()
    plt.savefig(output)


def save_summarize(metrics, filename, output):
    utils.save_metrics(filename, output, metrics)


def main():
    parser = argparse.ArgumentParser(description='Summarization algorithms.')
    parser.add_argument(
        'video', help='Name of the input video: videos/news.mpg')
    parser.add_argument(
        'difference', help='Name of difference: (pixels, blocks, histograms, edges.')
    parser.add_argument(
        '--gray', help='Folder to save summarized video.', default='./videos/gray')
    parser.add_argument(
        '--out', help='Folder to save summarized video.', default='./videos/out')

    args = parser.parse_args()

    video_name = args.video.split('/')[-1].rsplit('.', 1)[0]
    video_gray = '{}/{}.avi'.format(args.gray, video_name)
    video_out = '{}/{}_{}'.format(args.out, video_name, args.difference)

    metrics, peaks = process(video_gray, 64, args.difference)

    save_metrics(metrics, peaks, video_out + '.png')
    save_summarize(peaks, args.video, video_out + '.avi')


if __name__ == "__main__":
    main()
