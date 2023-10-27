import cv2
import numpy as np

from mobile_sam import sam_model_registry, SamPredictor
from utils import imshow, plot_circles, overlay


device = 'cpu'
image_size = (640, 480)


# Prepare MobileSAM Model
model_type = "vit_t"
sam_checkpoints = "./mobile_sam.pt"
mobile_sam = sam_model_registry[model_type](checkpoint=sam_checkpoints)
mobile_sam = mobile_sam.to(device)
mobile_sam.eval()
predictor = SamPredictor(mobile_sam)

# Prepare Anchor Image
anchor_image = cv2.imread("./images/anchor_image3.jpg")
anchor_image = cv2.resize(anchor_image, image_size)
interested_points = [[380, 300]]
anchor_image_circles = plot_circles(anchor_image.copy(), interested_points)
imshow("Anchor Image with circle", anchor_image_circles)
imshow("Anchor Image", anchor_image)

# Get Mask
anchor_points = np.array(interested_points)
anchor_labels = np.ones(len(anchor_points))
predictor.set_image(anchor_image, "BGR")
masks, scores, logits = predictor.predict(point_coords=anchor_points,
                                         point_labels=anchor_labels)
anchor_mask = masks[np.argmax(scores)]

masked_anchor_image = overlay(anchor_image, anchor_mask)
imshow("Masked Image", masked_anchor_image)