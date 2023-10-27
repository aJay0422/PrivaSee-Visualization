import cv2
import numpy as np
import time
import torch

from ultralytics import YOLO

from framemask import overlay


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

model = YOLO("yolov8n-seg.pt")
device = torch.device("cuda"
                      if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available()
                      else "cpu")

target_points = np.array([[0.5, 0.5]])  # in scale of the frame size


while True:
    starttime = time.time()
    # Capture frame by frame
    ret, frame = cap.read()
    frame = cv2.resize(frame, (640, 384))
    capturetime = time.time()
    print("Frame Capture Time: {:.2f}ms".format((capturetime - starttime) * 1000))

    mask_result = model(frame, device="cpu")
    inferencetime = time.time()
    print("Mask Inference Time: {:.2f}ms".format((inferencetime - capturetime) * 1000))
    masks = mask_result[0].masks

    if not masks:
        cv2.imshow("Camera", frame)
        if cv2.waitKey(1) == ord('q'):
            break
        continue
    mask_shape = masks[0].shape[1:]
    framesize = (mask_shape[1], mask_shape[0])

    # masks_overlay = []
    for i in range(len(masks)):
        mask = masks[i].data
        for target_point in target_points:
            point = [0, 0]
            point[0] = int(target_point[0] * mask_shape[1])
            point[1] = int(target_point[1] * mask_shape[0])
            if mask[0, point[1], point[0]]:
                # masks_overlay.append(mask)
                mask = np.squeeze(mask.data.cpu().numpy().astype(bool))
                frame = overlay(frame, mask)
                break

    # for mask in masks_overlay:
    #     mask = np.squeeze(mask.data.cpu().numpy().astype(bool))
    #     frame = cv2.resize(frame, (mask.shape[1], mask.shape[0]))
    #     frame = overlay(frame, mask)

    # Overlay points
    for point in target_points:
        frame = cv2.circle(frame, (int(framesize[0] * point[0]), int(framesize[1] * point[1])), radius=3,
                           color=(0, 255, 0), thickness=-1)

    overlaytime = time.time()
    print("Mask Overlay Time: {:.2f}ms".format((overlaytime - inferencetime) * 1000))

    if not ret:
        print("Can't receive frame")
        break

    # calculate fps
    endtime = time.time()
    frame = cv2.putText(frame, "FPS: {}".format(1 / (endtime - starttime)), org=(50, 50),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                        color=(255, 0, 0), thickness=2)

    print("Total Time: {:.2f}ms".format((endtime - starttime) * 1000))
    print("FPS: {:.1f}".format(1 / (endtime - starttime)))

    # Display the frame
    cv2.imshow("Camera", frame)
    if cv2.waitKey(1) == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()