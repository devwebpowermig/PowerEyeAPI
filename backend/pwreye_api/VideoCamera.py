import numpy as np
import cv2 as cv


def VideoCamera(self, *args, **kwargs):
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream ended?). Exiting...")
            break
        # Frame operations
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # Display resulting frame
        cv.imshow('frame', gray)
        if cv.waitKey(1) == ord('q'):
            break

    # Cleaning up
    cap.release()
    cv.destroyAllWindows()
