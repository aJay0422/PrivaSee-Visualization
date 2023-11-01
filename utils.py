import cv2
import numpy as np


def imshow(window_name, image):
    cv2.imshow(window_name, image)
    while True:
        k = cv2.waitKey(0) & 0xff
        if k == ord('q'):
            break


def overlay(image, mask, color=(0, 255, 0), alpha=0.5):
    """
    Overlay the mask on the image
    :param image: the image of size (H, W, 3)
    :param mask: the mask of size (H, W)
    :param alpha: transparency, a value between [0,1]
    :return: the masked image
    """
    mask_bool = np.squeeze(mask)
    image = cv2.resize(image, (mask_bool.shape[1], mask_bool.shape[0]))
    colored_mask = np.zeros_like(image)   # (H, W, 3)
    colored_mask[mask_bool] = color

    mask_area = image.copy()
    mask_area[~mask_bool] = 0
    nonmask_area = image.copy()
    nonmask_area[mask_bool] = 0

    alpha = 0.5
    mask_area = cv2.addWeighted(mask_area, 1-alpha, colored_mask, alpha, 0)
    image = mask_area + nonmask_area

    return image


def plot_circles(image, circles):
    for circle in circles:
        image = cv2.circle(image, (circle[0], circle[1]), radius=10,
                           color=(0, 0, 255), thickness=4)

    return image


def plot_cross(image, coord, length=4):
    x, y = coord
    image = cv2.line(image, (x - length, y), (x + length, y), (0, 0, 0), 2)
    image = cv2.line(image, (x, y - length), (x, y + length), (0, 0, 0), 2)
    return image


def overlay_icon(frame, icon, alpha, upperleft):
    icon_size = icon.shape[1::-1]
    x, y = upperleft[0], upperleft[1]
    frame[y: y + icon_size[1], x: x + icon_size[0], :] = \
        (frame[y: y + icon_size[1], x: x + icon_size[0], :] * (1 - alpha) +
         icon * alpha)
    return frame


def trans_decompose(img):
    img, img_alpha = img[:,:, :3], img[:, :, 3:]
    img_alpha = img_alpha / 255.0
    return img, img_alpha


def overlay_png(frame, img, coord, corner="ul", alpha=1.0):
    """
    Overlay a png image on the frame
    :param frame: the frame to be overlayed
    :param img: the png image to be overlayed, with alpha channel
    :param coord: the corner coordinate of the image
    :param corner: the corner type of the image to be overlayed, 
        can be 'ul', 'ur', 'll', 'lr' 
    :return: the frame with the png image overlayed
    """
    img_bgr, img_alpha = img[:, :, :3], img[:, :, 3:]
    img_alpha = img_alpha / 255.0 * alpha
    img_size = img_bgr.shape[1::-1]
    x, y = coord[0], coord[1]
    if corner == "ul":
        frame[y: y + img_size[1], x: x + img_size[0], :] = \
            (frame[y: y + img_size[1], x: x + img_size[0], :] * (1 - img_alpha) +
             img_bgr * img_alpha)
    elif corner == "ur":
        frame[y: y + img_size[1], x - img_size[0]: x, :] = \
            (frame[y: y + img_size[1], x - img_size[0]: x, :] * (1 - img_alpha) +
             img_bgr * img_alpha)
    elif corner == "ll":
        frame[y - img_size[1]: y, x: x + img_size[0], :] = \
            (frame[y - img_size[1]: y, x: x + img_size[0], :] * (1 - img_alpha) +
             img_bgr * img_alpha)
    elif corner == "lr":
        frame[y - img_size[1]: y, x - img_size[0]: x, :] = \
            (frame[y - img_size[1]: y, x - img_size[0]: x, :] * (1 - img_alpha) +
             img_bgr * img_alpha)
    else:
        raise ValueError("corner can only be 'ul', 'ur', 'll', 'lr'")



    return frame


def check_inbox(x, y, xyxy):
    x1, y1, x2, y2 = xyxy
    if x1 <= x <= x2 and y1 <= y <= y2:
        return True
    else:
        return False
