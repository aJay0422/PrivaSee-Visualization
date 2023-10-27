import cv2
import numpy as np
import time

import torch

from utils import overlay, imshow, plot_circles, plot_cross
from mobile_sam import sam_model_registry, SamPredictor
from LightGlue.lightglue import LightGlue, SuperPoint
from LightGlue.lightglue.utils import rbd


device = "cuda"

# Prepare MobileSAM Model
print("------Preparing Segmentation Model------")
model_type = 'vit_t'
sam_checkpoint = "./mobile_sam.pt"
mobile_sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
mobile_sam = mobile_sam.to(device=device)
mobile_sam.eval()
predictor = SamPredictor(mobile_sam)

# Prepare keypoints extractor and matcher
print("------Preparing keypoint extractor and matcher------")
extractor = SuperPoint(max_num_keypoints=1024).eval().cuda()
matcher = LightGlue(features="superpoint").eval().cuda()

# Real Time Segmentation
cap = cv2.VideoCapture(0)
matching_mode = False
ref_feats = None
ref_mask = None
while True:
    starttime = time.time()
    ret, frame = cap.read()
    image_size = (640, 480)
    frame = cv2.resize(frame, image_size)  # 480p

    if matching_mode:
        # Enter matching mode
        if ref_feats is None or ref_mask is None:
            ref_frame_tensor = torch.from_numpy(frame)
            ref_frame_tensor = torch.permute(ref_frame_tensor, (2, 0, 1)) / 255.0
            ref_frame_tensor = ref_frame_tensor.to(device)
            ref_feats = extractor.extract(ref_frame_tensor)

            input_points = np.array([[int(image_size[0] / 2), int(image_size[1] / 2)]])
            input_labels = np.array([1])
            predictor.set_image(frame, "BGR")
            masks, scores, logits = predictor.predict(point_coords=input_points,
                                                      point_labels=input_labels)
            ref_mask = masks[np.argmax(scores)]            
            continue
        else:
            matchstart = time.time()
            # extract target frame keypoints
            tar_frame_tensor = torch.from_numpy(frame)
            tar_frame_tensor = torch.permute(tar_frame_tensor, (2, 0, 1)) / 255.0
            tar_frame_tensor = tar_frame_tensor.to(device)
            tar_feats = extractor.extract(tar_frame_tensor)

            # match between ref and tar
            matches = matcher({"image0": ref_feats, "image1": tar_feats})
            ref_feats_tmp, tar_feats_tmp, matches = [rbd(x) for x in [ref_feats, tar_feats, matches]]
            matches = matches["matches"]
            ref_points = ref_feats_tmp["keypoints"][matches[..., 0]]
            tar_points = tar_feats_tmp["keypoints"][matches[..., 1]]

            # check whether the target shows
            shown = False
            input_points = []
            input_labels = []
            for i in range(len(tar_points)):
                coord_ref = [int(ref_points[i, 0].item()), int(ref_points[i, 1].item())]
                if ref_mask[coord_ref[1], coord_ref[0]]:
                    shown = True
                    input_points.append([int(tar_points[i, 0].item()), int(tar_points[i, 1].item())])
                    input_labels.append(1)
            input_points = np.array(input_points)
            input_labels = np.array(input_labels)
            matchend = time.time()
            print("Matching Time: {:.2f}".format(matchend - matchstart))


            if shown:
                inf_start = time.time()
                predictor.set_image(frame)
                masks, scores, logits = predictor.predict(point_coords=input_points,
                                                          point_labels=input_labels)
                mask = masks[np.argmax(scores)]
                inf_end = time.time()
                print("Inference Time: {:.2f}".format(inf_end - inf_start))


    else:
        # prompt
        input_points = np.array([[int(image_size[0] / 2), int(image_size[1] / 2)]])
        input_labels = np.array([1])

        # Inference
        inf_start = time.time()
        predictor.set_image(frame, "BGR")
        masks, scores, logits = predictor.predict(point_coords=input_points,
                                                point_labels=input_labels)
        mask = masks[np.argmax(scores)]
        inf_end = time.time()
        print("Inference Time: {:.2f}".format(inf_end - inf_start))

    # Overlay mask
    ovl_start = time.time()
    masked_frame = overlay(frame, mask)
    masked_frame = plot_cross(masked_frame, (int(image_size[0] / 2), int(image_size[1] / 2), 5))
    masked_frame = plot_circles(masked_frame, input_points)
    ovl_end = time.time()
    print("Overlay Time: {:.2f}".format(ovl_end - ovl_start))

    # Display
    cv2.imshow("Camera", masked_frame)
    endtime = time.time()
    fps = 1 / (endtime - starttime)
    print("FPS: {:.1f}".format(fps))
    k = cv2.waitKey(1)
    if k == ord('q'):
        break
    elif k == ord('n'):
        # Anchor the next image
        matching_mode = True
