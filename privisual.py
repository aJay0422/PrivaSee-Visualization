import cv2
import numpy as np

from mobile_sam import sam_model_registry, SamPredictor
from utils import imshow, overlay_icon


device = "cpu"
image_size = (1280, 960)
icon_y = 100

# Prepare icon
icon = cv2.imread("./images/icons/range1.png", cv2.IMREAD_UNCHANGED)
icon_size = icon.shape[:2]
icon_size = (int(icon_size[1] / icon_size[0] * icon_y), icon_y)
icon = cv2.resize(icon, icon_size)
alpha = icon[:, :, 3:] / 255.0
icon = icon[:, :, :3]
upperleft = (600, 600)

# VideoCapture
cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    frame = overlay_icon(frame, icon, alpha, upperleft)
    cv2.imshow("Camera", frame)
    k = cv2.waitKey(1)
    if k == ord('q'):
        break