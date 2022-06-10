import cv2 as cv
import argparse
import os


def to_grayscale(input_name, output_name):
    cap = cv.VideoCapture(input_name)

    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv.VideoWriter_fourcc(*'XVID')
    out = cv.VideoWriter(output_name, fourcc, 20.0,
                         (width, height), isColor=False)

    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            out.write(gray_frame)
        else:
            cap.release()
            out.release()
            break


def bgr_to_grayscale(files, output_path='./videos/gray'):
    for i in range(len(files)):
        output_name = '{}/{}'.format(output_path,
                                     files[i].split('/')[-1].rsplit('.', 1)[0] + '.avi')
        to_grayscale(files[i], output_name)
        print(files[i], output_name)


def main():
    path_color = './videos/color'
    path_gray = './videos/gray'
    path_out = './videos/out'

    if not os.path.exists(path_color):
        os.makedirs(path_color)
    if not os.path.exists(path_gray):
        os.makedirs(path_gray)
    if not os.path.exists(path_out):
        os.makedirs(path_out)

    parser = argparse.ArgumentParser(
        description='Convert videos to grayscale.')
    parser.add_argument(
        'video', help='Name of the input video: videos/news.mpg')

    args = parser.parse_args()

    if args.video == 'all':
        files = os.listdir('./videos/color')
        files = ['{}/{}'.format(path_color, files[i])
                 for i in range(len(files))]
        bgr_to_grayscale(files)
    else:
        bgr_to_grayscale([args.video])


if __name__ == "__main__":
    main()
