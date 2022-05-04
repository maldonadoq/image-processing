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


def main():
    parser = argparse.ArgumentParser(description='Halftones algorithm.')
    parser.add_argument(
        'in_image', help='Name of the input image: images/baboon.png')
    parser.add_argument(
        'in_distr', help='Name of error distribution: (floyd, stevenson, burkes, sierra, stucki, jarvis).')
    parser.add_argument(
        '--zigzag', help='Use zigzag alternating.', action='store_true')
    parser.add_argument(
        '--color', help='Use color images or grayscale.', action='store_true')
    args = parser.parse_args()


if __name__ == "__main__":
    main()
