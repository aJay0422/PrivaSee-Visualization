# from segment_anything import SamPredictor, sam_model_registry
from utils import overlay, imshow
import cv2
import numpy as np
import time
from tqdm import tqdm
import torch
from ultralytics import YOLO
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pixellib
from pixellib.torchbackend.instance import instanceSegmentation

from FastSAM.fastsam import FastSAM, FastSAMPrompt
from mobile_sam import sam_model_registry, SamPredictor


device = torch.device("cuda"
                      if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available()
                      else "cpu")

checkpoint_folder = "./sam_checkpoints/"
checkpoint_paths = {
    "vit_b": checkpoint_folder + "sam_vit_b_01ec64.pth",
    "vit_h": checkpoint_folder + "sam_vit_h_4b8939.pth",
    "vit_l": checkpoint_folder + "sam_vit_l_0b3195.pth"
}



class FrameMasker():
    def __init__(self, model="vit_b"):
        """
        Initialize a frame masker
        :param model: Name of the image encoder, can be "vit_b", "vit_l" or "vit_h"
            vit_b: 375MB; vit_l: 1.25GB; vit_h: 2.56GB
        """
        assert model in ["vit_b", "vit_h", "vit_l"], "model should be vit_b, vit_h or vit_l"

        self.checkpoint_path = checkpoint_paths[model]
        self.sam = sam_model_registry[model](checkpoint=self.checkpoint_path)
        # self.sam.to(device)
        self.predictor = SamPredictor(self.sam)
        self.colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0),
                       (0,255, 255), (255, 0, 255), (255, 255, 0)]

    def predict(self, image, coords, labels, alpha=0.5):
        """
        Get masks and overlay them to the image
        :param image: the image needed to be segmented
        :param coords: A Nx2 array of points coordinatesin (X, Y) in pixels
        :param labels: A length N array of labels for the point. 1 indicates a foreground point and 0 indicates a
            background point
        :param alpha: A value between [0, 1], indicating the transparency level of the mask. 1 means no transparency,
            0 means completely transparent
        :return: A masked image with the same size of the input image.
        """

        starttime = time.time()
        self.predictor.set_image(image)   # set the BGR image for prediction
        setimagetime = time.time()

        # generate one mask for each coordinate
        for i, (coord, label) in enumerate(zip(coords, labels)):
            # generate a boolean mask for this coordinate
            masks, scores, logits = self.predictor.predict(point_coords=np.array([coord]),
                                                           point_labels=np.array([label]),
                                                           multimask_output=True)
            idx = np.argmax(scores)   # pick the mask with maximum confidence score
            mask_bool = masks[idx]

            # get a colored mask
            color = self.colors[i]   # B G R
            colored_mask = np.zeros_like(image)
            colored_mask[mask_bool] = color

            # separate mask area and non-mask area from the input image
            mask_area = image.copy()
            mask_area[~mask_bool] = 0
            nonmask_area = image.copy()
            nonmask_area[mask_bool] = 0

            # overlay the mask onto the mask area
            mask_area = cv2.addWeighted(mask_area, 1 - alpha, colored_mask, alpha, 0)

            # add mask area and non-mask area
            image = mask_area + nonmask_area

        inferencetime = time.time()

        print("Set Image time: {}".format(setimagetime - starttime))
        print("Inference time: {}".format(inferencetime - setimagetime))

        return image   # BGR Image with shape (H, W, 3)


def get_masker(model="vit_b"):
    return FrameMasker(model=model)



def test1():
    image = cv2.imread("./images/table1.jpeg")
    input_points = np.array([[600, 600]])
    input_labels = np.array([1])

    masker = FrameMasker(model="vit_b")
    starttime = time.time()
    for i in tqdm(range(1)):
        masked_image = masker.predict(image, input_points, input_labels)
    endtime =time.time()

    print("Average time: {:.2f}".format((endtime - starttime) / 1))

    cv2.imshow("Image", image)
    cv2.imshow("Masked Image", masked_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return masked_image

def test2():
    image = cv2.imread("./images/table2.jpeg")
    input_points = np.array([[800, 350], [200, 500]])
    input_labels = np.array([1, 1])

    masker = FrameMasker(model="vit_b")
    masked_image = masker.predict(image, input_points, input_labels)

    cv2.imshow("Image", image)
    cv2.imshow("Masked Image", masked_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return masked_image


def test3():
    # MediaPipe
    # Load Image
    image = mp.Image.create_from_file("./images/table1.jpeg")

    # Inference
    BG_COLOR = (192, 192, 192)   # gray
    MASK_COLOR = (255, 255, 255)   # white

    base_options = python.BaseOptions(model_asset_path="./deeplab_v3.tflite")
    options = vision.ImageSegmenterOptions(base_options=base_options)

    # create teh image segmenter
    with vision.ImageSegmenter.create_from_options(options) as segmenter:
        segmentation_result = segmenter.segment(image)
        category_mask = segmentation_result.category_mask

        image_data = image.numpy_view()
        fg_image = np.zeros(image_data.shape, dtype=np.uint8)
        fg_image[:] = MASK_COLOR
        bg_image = np.zeros(image_data.shape, dtype=np.uint8)
        bg_image[:] = BG_COLOR

        condition = np.stack((category_mask.numpy_view(),) * 3, axis=-1) > 0.2
        output_image = np.where(condition, fg_image, bg_image)

        print("Segmentation mask")


def test4():
    # Pixellib
    ins = instanceSegmentation()
    ins.load_model("./pointrend_resnet50.pkl")
    ins.segmentImage("./images/table1.jpeg", show_bboxes=True, output_image_name="output_image.jpg")


def test5():
    model = YOLO("yolov8n-seg.pt")

    image = cv2.imread("./images/table1.jpeg")
    results = model(image)


def test6():
    # FastSAM

    model_path = "./FastSAM.pt"
    model = FastSAM(model_path)

    point_prompt = np.array([[600, 600]])
    point_label = np.array([1])

    input = cv2.imread("./images/table1.jpeg")
    input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available()
                          else "cpu")
    retina = False   # whether high resulotion
    imgsz = 720
    conf = 0.4
    iou = 0.9
    starttime = time.time()
    everything_results = model(
        input,
        device=device,
        retina_masks=retina,
        imgsz=imgsz,
        conf=conf,
        iou=iou
    )

    prompt_process = FastSAMPrompt(input, everything_results, device=device)
    ann = prompt_process.point_prompt(
        points=point_prompt, pointlabel=point_label
    )
    # points = point_prompt
    # prompt_process.plot(
    #     annotations=ann,
    #     output_path="./table1.jpeg",
    #     bboxes=None,
    #     points=points,
    #     point_label=point_label,
    #     withContours=False,
    #     better_quality=False
    # )
    inferencetime = time.time()
    print("Inference Time: {}".format(inferencetime - starttime))

    masked_image = overlay(input, ann)
    cv2.imshow("masked image", masked_image)
    cv2.waitKey(0)


def test7():
    # sam
    # Load model
    device = torch.device("cuda"
                      if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available()
                      else "cpu")
    sam = sam_model_registry["vit_b"](checkpoint="./sam_checkpoints/"+"sam_vit_b_01ec64.pth")
    sam.to(device)
    predictor = SamPredictor(sam)

    # Load image
    image = cv2.imread("./images/table1.jpeg")

    # Set image
    starttime = time.time()
    predictor.set_image(image)
    setimagetime = time.time()

    # Target Point
    point = np.array([[600, 300]])
    label = np.array([[1]])

    # Inference
    masks, scores, logits = predictor.predict(
        point_coords=np.array(point),
        point_labels=np.array(label),
        multimask_output=True
    )
    idx = np.argmax(scores)
    mask_bool = masks[idx]

    # overlay image
    masked_image = overlay(image, mask_bool)

    # show image
    cv2.imshow("Masked Image", masked_image)
    cv2.waitKey(0)


def test8():
    # Mobile SAM
    model_type = 'vit_t'
    sam_checkpoint = "./mobile_sam.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available()
                          else "cpu")
    device = "cpu"

    mobile_sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    mobile_sam.to(device=device)
    mobile_sam.eval()

    image = cv2.imread("./images/table1.jpeg")
    input_points = np.array([[600, 600]])
    input_labels = np.array([1])

    predictor = SamPredictor(mobile_sam)
    starttime = time.time()
    for i in range(10):
        predictor.set_image(image)
        masks, _, _ = predictor.predict(point_coords=input_points,
                                        point_labels=input_labels)
    print("Total time: {}".format((time.time() - starttime) / 10))
    masked_image = overlay(image, masks[0])
    imshow("masked", masked_image)
    pass

    



if __name__ == "__main__":
    test8()

