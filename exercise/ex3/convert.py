import cv2 as cv
import os


def to_grayscale(input_name, output_name):
    cap = cv.VideoCapture(input_name)

    print(cv.CAP_PROP_FRAME_WIDTH)
    print(cv.CAP_PROP_FRAME_HEIGHT)
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv.VideoWriter_fourcc(*'DIVX')
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


def bgr_to_grayscale(input_path='./videos/color', output_path='./videos/gray'):
    files = os.listdir(input_path)

    for i in range(len(files)):
        input_name = '{}/{}'.format(input_path, files[i])
        output_name = '{}/{}'.format(output_path,
                                     files[i].rsplit('.')[0] + '.mp4')
        print(input_name, output_name)
        to_grayscale(input_name, output_name)


def main():
    bgr_to_grayscale()


if __name__ == "__main__":
    main()
