from matplotlib import pyplot as plt

import numpy as np
import argparse
import utils
import time
import os


def diff_pixels(frames, params):
    # diff between two consecutive frames in a shot
    difference = np.abs(frames[1:] - frames[:-1])

    # number of pixels greater than a threshold
    metrics = np.sum(difference > params[0], axis=(1, 2))

    return list(metrics)


def diff_blocks(frames, params):
    # parmas[0] is block size
    rows, cols = frames.shape[1], frames.shape[2]

    # trows is number os blocks
    trows = rows // params[0]
    tcols = cols // params[0]

    # use new frame dimensions
    # diff between two consecutive frames
    # squared difference
    frames = frames[:, :trows * params[0], :tcols * params[0]]
    difference = np.abs(frames[1:] - frames[:-1])
    difference = difference * difference

    metrics = []
    for diff in difference:
        # vectorized sum blocks
        # vectorized sqrt soma
        soma = np.sum(utils.reshape(
            diff, trows, tcols, params[0]), axis=(1, 2))
        metrics.append((np.sqrt(soma) > params[1]).sum())
    return metrics


def diff_histogram(frames, params):
    total = frames.shape[0]
    # array of histograms
    histograms = np.empty((total, 256))

    # numpy histograms
    for i in range(total):
        hist, _ = np.histogram(frames[i], bins=256, range=(0, 255))
        histograms[i] = hist

    # diff between two consecutive frames
    # T = mean + 3*std
    difference = np.abs(histograms[1:] - histograms[:-1])
    means = np.mean(difference, axis=1)
    stds = np.std(difference, axis=1)
    metrics = means + stds*3

    return list(metrics)


def diff_edges(frames, params):
    metrics = []
    for frame in frames:
        # apply sobel filter and count pixels
        # sobel filter is vectorized as addition
        mask = utils.normalize(utils.sobel(frame))
        metrics.append((mask > params[0]).sum())
    return metrics


def process(color_video, gray_video, shot_size, diff):
    filename = gray_video
    get_metrics = utils.get_metrics_gray
    if not os.path.exists(gray_video):
        filename = color_video
        get_metrics = utils.get_metrics_color

    metrics = None
    # summarize video using specific difference
    if diff == 'pixels':
        metrics = get_metrics(filename, shot_size, diff_pixels, [128], 'int16')
    elif diff == 'blocks':
        metrics = get_metrics(filename, shot_size,
                              diff_blocks, [8, 128], 'int32')
    elif diff == 'histograms':
        metrics = get_metrics(filename, shot_size, diff_histogram, [])
    elif diff == 'edges':
        metrics = get_metrics(filename, shot_size, diff_edges, [128])
    else:
        raise NameError("'{}' > not defined".format(diff))

    return metrics


def save_metrics(metrics, peaks, output):
    X = np.array(range(len(metrics)))

    plt.figure(figsize=(25, 10))
    index = peaks.nonzero()[0]
    plt.plot(X[index], metrics[index], "o", label="peaks")
    plt.plot(X, metrics, label="metrics")
    plt.legend()
    plt.savefig(output, bbox_inches='tight')


def save_summarize(metrics, filename, output):
    utils.save_metrics(filename, output, metrics)


def main():
    parser = argparse.ArgumentParser(description='Summarization algorithms.')
    parser.add_argument(
        'video', help='Name of the input video: videos/news.mpg')
    parser.add_argument(
        'difference', help='Name of difference: (pixels, blocks, histograms, edges.')
    parser.add_argument(
        '--gray', help='Folder to save summarized video. (./videos/gray)', default='./videos/gray')
    parser.add_argument(
        '--out', help='Folder to save summarized video. (./videos/out)', default='./videos/out')

    args = parser.parse_args()

    video_name = args.video.split('/')[-1].rsplit('.', 1)[0]
    video_gray = '{}/{}.avi'.format(args.gray, video_name)
    video_out = '{}/{}_{}'.format(args.out, video_name, args.difference)

    start = time.time()
    metrics, peaks = process(args.video, video_gray, 64, args.difference)
    end = time.time()
    print('{} - {}, summarized: {}, time: {}'.format(video_name,
          args.difference, peaks.sum() / metrics.shape[0], end - start))

    save_metrics(metrics, peaks, video_out + '.png')
    save_summarize(peaks, args.video, video_out + '.avi')


if __name__ == "__main__":
    main()

# python3 summarize.py ./videos/color/news.mpg edges
