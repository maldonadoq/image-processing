import cv2 as cv


def save_metrics(file, output, metrics):
    cap = cv.VideoCapture(file)
    print(file, output)

    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv.VideoWriter_fourcc(*'XVID')
    out = cv.VideoWriter(output, fourcc, 20.0, (width,  height))

    count = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            if count < len(metrics) and metrics[count]:
                out.write(frame)
        else:
            cap.release()
            out.release()
            break
        count += 1
