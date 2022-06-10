
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


def save_metrics(metrics, output):
    X = range(len(metrics))

    plt.figure(figsize=(25, 10))
    plt.plot(X, metrics)
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
        '--folder', help='Folder to save summarized video.', default='./videos')

    args = parser.parse_args()

    metrics = process(args.video, 64, args.difference)
    path = '{}_{}'.format(args.video.rsplit('.', 1)[0], args.difference)

    save_metrics(metrics, path + '.png')
    save_summarize(metrics, args.video, path + '.avi')


if __name__ == "__main__":
    main()
