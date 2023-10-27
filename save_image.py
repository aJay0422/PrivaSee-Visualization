import cv2
from utils import imshow, trans_decompose, overlay_icon

def save1():
    image_size = (1280, 960)
    img = cv2.imread("./images/test/anchor_vib1.jpg")
    img = cv2.resize(img, image_size)

    tag_y = 250
    tag = cv2.imread("./images/icons/prop_tag1.png", cv2.IMREAD_UNCHANGED)
    tag_size = tag.shape[1::-1]
    tag = cv2.resize(tag, (int(tag_y * tag_size[0] / tag_size[1]), tag_y))
    tag, tag_alpha = trans_decompose(tag)
    img = overlay_icon(img, tag, tag_alpha, (500, 330))
    imshow("img", img)



if __name__ == "__main__":
    save1()