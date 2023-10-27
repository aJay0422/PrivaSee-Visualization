import cv2
import time


def camera_test():
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    while True:
        starttime = time.time()
        ret, frame = cap.read()

        endtime = time.time()

        fps = 1 / (endtime - starttime)
        print("FPS: {:.4f}".format(fps))
        frame = cv2.putText(frame, "FPS: {:.2f}".format(fps), [10, 30],
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Camera", frame)
        k = cv2.waitKey(1)
        if k == ord('q'):
            break


if __name__ == "__main__":
    camera_test()