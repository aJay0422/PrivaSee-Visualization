########################################
# Segmentation with prior anchor image #
########################################

import cv2
import numpy as np
import time
import torch

from utils import overlay, plot_circles
from mobile_sam import sam_model_registry, SamPredictor
from LightGlue.lightglue import LightGlue, SuperPoint
from LightGlue.lightglue.utils import rbd


device = "cpu"
image_size = (640, 480)

# Prepare MobileSAM Model
print("----------Preparing Segmentation Model----------")
model_type = "vit_t"
sam_checkpoints = "./mobile_sam.pt"
mobile_sam = sam_model_registry[model_type](checkpoint=sam_checkpoints)
mobile_sam = mobile_sam.to(device)
mobile_sam.eval()
predictor = SamPredictor(mobile_sam)

# Prepare keypoints extractor and matcher
print("----------Preparing keypoint extractor and matcher---------")
extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)
matcher = LightGlue(features="superpoint").eval().to(device)

# Prepare Anchor Image
anchor_path = "./images/anchor_image.png"
anchor_image = cv2.imread(anchor_path)
anchor_image = cv2.resize(anchor_image, image_size)
anchor_image_tensor = torch.from_numpy(anchor_image)
anchor_image_tensor = torch.permute(anchor_image_tensor, (2, 0, 1)) / 255.0
anchor_image_tensor = anchor_image_tensor.to(device)
anchor_feats = extractor.extract(anchor_image_tensor)
anchor_points = np.array([[int(image_size[0] / 2), int(image_size[1] / 2)]])
anchor_labels = np.array([1])
predictor.set_image(anchor_image, "BGR")
masks, scores, logits = predictor.predict(point_coords=anchor_points,
                                          point_labels=anchor_labels)
anchor_mask = masks[np.argmax(scores)]


# Start read frames
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    # read frame-by-frame
    starttime = time.time()
    ret, frame = cap.read()

    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame. Exiting....")
        break

    # Normalize image size
    frame = cv2.resize(frame, image_size)   # 480p

    # extract target frame keypoints
    extstart = time.time()
    frame_tensor = torch.from_numpy(frame)
    frame_tensor = torch.permute(frame_tensor, (2, 0, 1)) / 255.0
    frame_tensor = frame_tensor.to(device)
    frame_feats = extractor.extract(frame_tensor)
    extend = time.time()
    print("Extract Time: {:.3f}".format(extend - extstart))

    # match between current frame and anchor image
    matstart = time.time()
    matches = matcher({"image0": anchor_feats, "image1": frame_feats})
    anchor_feats_tmp, frame_feats_tmp, matches = [rbd(x) for x in [anchor_feats, frame_feats, matches]]
    matches = matches["matches"]
    anchor_points = anchor_feats_tmp["keypoints"][matches[..., 0]]
    frame_points = frame_feats_tmp["keypoints"][matches[..., 0]]

    # check whether the target shows
    shown = False
    input_points = []
    input_labels = []
    for i in range(len(frame_points)):
        coord_anchor = [int(anchor_points[i, 0].item()), int(anchor_points[i, 1].item())]
        if anchor_mask[coord_anchor[1], coord_anchor[0]]:
            shown = True
            input_points.append([int(frame_points[i, 0].item()), int(frame_points[i, 1].item())])
            input_labels.append(1)
    input_points = np.array(input_points)
    input_labels = np.array(input_labels)
    matend = time.time()
    print("Matching Time: {:.3f}".format(matend - matstart))

    if not shown:
        masked_frame = frame
    else:
        infstart = time.time()
        predictor.set_image(frame)
        masks, scores, logits = predictor.predict(point_coords=input_points,
                                                  point_labels=input_labels)
        mask = masks[np.argmax(scores)]
        infend = time.time()
        print("Inference Time: {:.3f}".format(infend - infstart))

        # Overlay mask
        masked_frame = overlay(frame, mask)
        masked_frame = plot_circles(masked_frame, input_points)

    # Display
    endtime = time.time()
    fps = 1 / (endtime - starttime)
    masked_frame = cv2.putText(masked_frame, "FPS: {:.2f}".format(fps), [10, 30],
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("Camrea", masked_frame)
    k = cv2.waitKey(1)
    if k == ord('q'):
        break
