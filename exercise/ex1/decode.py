import numpy as np
import cv2 as cv
import argparse


def steganography_decode(input_image, bit_plane):
    # Get k Least Significant Bit
    mask = 1 << bit_plane
    #input_bits = (input_image & mask).ravel() // bit_plane
    input_bits = (input_image & mask).ravel() // mask

    # Broadcast to array nx8 Dim
    output_bits = input_bits.reshape((-1, 8))
    # Sum axis=1 to get ascii value
    output_bytes = np.sum(output_bits * [128, 64, 32, 16, 8, 4, 2, 1], axis=1)

    output_text = ''
    for ascii in output_bytes:
        if ascii == 0:
            break
        output_text += chr(ascii)

    return output_text


def main():

    # Add arguments to encode steganography program
    parser = argparse.ArgumentParser(description='Steganography Decoder.')
    parser.add_argument('input_image', help='Name of the encoded image.')
    parser.add_argument('output_text', help='Name of the decoded text.')
    parser.add_argument('bit_plane', help='Bit plane to find msg.', type=int)
    args = parser.parse_args()

    # Read encoded image file
    input_image = cv.imread(args.input_image, cv.IMREAD_COLOR)

    output_text = steganography_decode(input_image, args.bit_plane)
    with open(args.output_text, 'w') as file:
        file.write(output_text)


if __name__ == "__main__":
    main()

# Example to tun program
# python3 decode.py images/baboon_enc.png data/small_dec.txt 0
