import cv2
from utils import imshow
from qreader import QReader



def read_qr_code(img):
    detect = QReader()
    text = detect.detect_and_decode(img)
    return text

cap = cv2.VideoCapture(1)
while True:
    ret, frame = cap.read()
    cv2.imshow("frame", frame)
    text = read_qr_code(frame)
    if len(text) == 0:
        print("empty")
    else:
        print(text)
    k = cv2.waitKey(1)
    if k == ord('q'):
        break

