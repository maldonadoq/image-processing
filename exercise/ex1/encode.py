import numpy as np
import cv2 as cv
import argparse


def steganography_encode(input_image, input_text, bit_plane):
    # Check if the message can be encoded in the image
    if(len(input_text) > (input_image.size // 8)):
        raise ValueError("Insufficient bytes")

    # Get the bits of the input text
    input_ascii = np.frombuffer(input_text.encode(), 'S1').view(np.uint8)
    input_bits = np.unpackbits(input_ascii)

    # Flatten input image array
    output_flat = input_image.ravel()
    output_use = output_flat[:len(input_bits)]

    # You can set [0,1]:
    # 1: x = x | pos
    # 0: x = x & ~pos
    mask_one = 1 << bit_plane
    mask_zero = ~mask_one

    output_flat[:len(input_bits)] = np.where(
        input_bits, output_use | mask_one, output_use & mask_zero)

    # Recover image shape
    output_image = output_flat.reshape(input_image.shape)
    return output_image


def main():

    # Add arguments to encode steganography program
    parser = argparse.ArgumentParser(description='Steganography Encoder.')
    parser.add_argument('in_image', help='Name of the input image.')
    parser.add_argument('in_text', help='Name of the input text.')
    parser.add_argument('out_image', help='Name of the encoded image.')
    parser.add_argument('bit_plane', help='Bit plane to hide msg.', type=int)
    args = parser.parse_args()

    # Read input image file
    input_image = cv.imread(args.in_image, cv.IMREAD_COLOR)

    # Read input text file
    input_text = ''
    with open(args.in_text, 'r') as file:
        input_text = file.read() + chr(0)

    output_image = steganography_encode(
        input_image, input_text, args.bit_plane)
    cv.imwrite(args.out_image, output_image)


if __name__ == "__main__":
    main()
