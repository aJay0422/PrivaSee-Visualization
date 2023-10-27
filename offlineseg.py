import cv2
import numpy as np
import time
import torch
from segment_anything import SamPredictor, sam_model_registry
from LightGlue.lightglue import LightGlue, SuperPoint, DISK
from LightGlue.lightglue.utils import rbd
from utils import imshow, overlay, plot_circles


## Show Reference Image
ref_image = cv2.imread("./reference_img.jpg")
# resize to (x, y) = (1080, 720)
ref_image = cv2.resize(ref_image, (1080, 720))
cv2.imshow("reference image", ref_image)
print(ref_image.shape)

while True:
    k = cv2.waitKey(0) & 0xff
    if k == ord("q"):
        cv2.destroyAllWindows()
        break


## Load SAM Model
checkpoint_path = "./sam_checkpoints/sam_vit_b_01ec64.pth"
sam = sam_model_registry["vit_b"](checkpoint=checkpoint_path)
predictor = SamPredictor(sam)


## Inference
target_points = np.array([[640, 300]])
target_labels = np.array([1])

predictor.set_image(ref_image)
masks, scores, logits = predictor.predict(point_coords=target_points,
                                          point_labels=target_labels,
                                          multimask_output=True)
idx = np.argmax(scores)
mask_bool = masks[idx]


masked_ref_image = overlay(ref_image, mask_bool)
# show masked image
cv2.imshow("Masked Reference Image", masked_ref_image)
while True:
    k = cv2.waitKey(0) & 0xff
    if k == ord("q"):
        cv2.destroyAllWindows()
        break


## Keypoints Extract
# SuperPoint + LightGlue
extractor = SuperPoint(max_num_keypoints=2048).eval()
matcher = LightGlue(features="superpoint").eval()
# extract local features
ref_image_tensor = torch.from_numpy(ref_image)
ref_image_tensor = torch.permute(ref_image_tensor, (2, 0, 1)) / 255.0   # (3, H, W) and normalize to [0, 1]
feats_ref = extractor.extract(ref_image_tensor)


## Plot Keypoints
keypoints = feats_ref['keypoints'].detach().cpu().squeeze().numpy().astype(int)
coords_plot = []
for i in range(len(keypoints)):
    coord = [keypoints[i, 1], keypoints[i, 0]]
    if mask_bool[coord[0], coord[1]]:
        coords_plot.append(coord)
for coord in coords_plot:
    masked_ref_image = cv2.circle(masked_ref_image, (coord[1], coord[0]), radius=3,
                                  color=(255, 0, 0), thickness=1)
cv2.imshow("Masked Reference Image with keypoints", masked_ref_image)
while True:
    k = cv2.waitKey(0) & 0xff
    if k == ord("q"):
        cv2.destroyAllWindows()
        break


## Capture Video
cap = cv2.VideoCapture("./target_video.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
# out = cv2.VideoWriter("./target_video_masked.avi", cv2.VideoWriter_fourcc('M','J','P','G'), fps, (1080,720))
mask_bool_ref = mask_bool
while cap.isOpened():
    ret, frame = cap.read()
    starttime = time.time()
    if ret == True:
        ## Feature Matching
        # extract target features
        frame = cv2.resize(frame, (1080, 720))
        target_image_tensor = torch.from_numpy(frame)
        target_image_tensor = torch.permute(target_image_tensor, (2, 0, 1)) / 255.0
        feats_tar = extractor.extract(target_image_tensor)

        # match between ref and tar
        matches = matcher({"image0": feats_ref, "image1": feats_tar})
        feats_ref_tmp, feats_tar_tmp, matches = [rbd(x) for x in [feats_ref, feats_tar, matches]]
        matches = matches["matches"]
        points_ref = feats_ref_tmp["keypoints"][matches[..., 0]]
        points_tar = feats_tar_tmp["keypoints"][matches[..., 1]]

        # check whether the target shows
        shown = False
        target_points = []
        target_labels = []
        ref_points = []
        for i in range(len(points_tar)):
            coord_ref = [int(points_ref[i, 0].item()), int(points_ref[i, 1].item())]
            if mask_bool_ref[coord_ref[1], coord_ref[0]]:
                shown = True
                ref_points.append([coord_ref[0], coord_ref[1]])
                target_points.append([int(points_tar[i, 0].item()), int(points_tar[i, 1].item())])
                target_labels.append(1)
        target_points = np.array(target_points)
        target_labels = np.array(target_labels)

        # If target shows, send it to SAM and get segmentation
        tmp_tar_image = plot_circles(frame, target_points)
        # tmp_ref_image = plot_circles(ref_image, ref_points)
        # imshow("Ref w. circle", tmp_ref_image)
        imshow("Tar w. cricle", tmp_tar_image)
        # imshow("Ref", ref_image)
        if shown:
            predictor.set_image(frame)
            masks, scores, logits = predictor.predict(point_coords=target_points,
                                                      point_labels=target_labels,
                                                      multimask_output=True)
            idx = np.argmax(scores)
            mask_bool = masks[idx]

            # overlay
            frame = overlay(frame, mask_bool)

        # out.write(frame)
        print("1 Frame written in {} seconds".format(int(time.time() - starttime)))

    else:
        break


## Capture Video 2
cap = cv2.VideoCapture("./target_video.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter("./target_video_masked2.avi")

mask_bool_ref = mask_bool

while cap.isOpened():
    ret, frame = cap.read()
    starttime = time.time()
    if ret == True:
        ## Feature Matching
        # extract target feature
        frame = cv2.resize(frame, (1080, 720))
        target_image_tensor = torch.from_numpy(frame)
        target_image_tensor = torch.permute(target_image_tensor, (2, 0, 1)) / 255.0

    else:
        break







pass
