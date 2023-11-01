import cv2

from utils import overlay_png, check_inbox


# Settings
frame_size = (1280, 720)
image_center = (int(frame_size[0] / 2), int(frame_size[1] / 2))

selection_ui_size = (600, 600)
selection_ui_coord = (image_center[0] - int(selection_ui_size[0] / 2),
                      image_center[1] - int(selection_ui_size[1] / 2))

icon_size = (90, 90)
sensor_icon_ul = (20, 20)

back_button_size = (150, 100)
back_button_ul = (frame_size[0] - back_button_size[0] - 20, 5)

property_icon_height = 90


# Preparation
## sensor selection UI
sensor_selection = cv2.imread("./images/icons/selection.png", cv2.IMREAD_UNCHANGED)
sensor_selection = cv2.resize(sensor_selection, selection_ui_size)
## sensor icons
sensor_list = ["vib1", "vib2", "mic1"]
sensor_icons = {}
for sensor_name in sensor_list:
    icon = cv2.imread("./images/icons/" + sensor_name + ".png", cv2.IMREAD_UNCHANGED)
    icon = cv2.resize(icon, icon_size)
    sensor_icons[sensor_name] = icon
sensor_icons_bbox_xyxy = {"vib1": (394, 456, 495, 538),
                          "vib2": (785, 457, 885, 540),
                          "mic1": (610, 100, 668, 180)}   # bbox in the selection UI
## back button
back_button = cv2.imread("./images/icons/backbutton.png", cv2.IMREAD_UNCHANGED)
back_button = cv2.resize(back_button, back_button_size)
back_button_bbox_xyxy = (back_button_ul[0], back_button_ul[1],
                         back_button_ul[0] + back_button_size[0],
                         back_button_ul[1] + back_button_size[1])
## property icons
property_icons_names = ["walking", "cooking", "exercising", "watching",
                        "identification1", "identification2", "identification3",
                        "localization1", "localization2", "localization3",
                        "sound1", "sound2", "sound3"]
property_icons = {}
for property_name in property_icons_names:
    property_icon = cv2.imread("./images/icons/" + property_name + ".png", cv2.IMREAD_UNCHANGED)
    temp_size = property_icon.shape[:2]
    temp_size = (int(property_icon_height / temp_size[1] * temp_size[0]), property_icon_height)
    property_icon = cv2.resize(property_icon, temp_size)
    property_icons[property_name] = property_icon





def draw_on_frame(frame, mode):
    if mode == "selection":
        frame = overlay_png(frame, sensor_selection, selection_ui_coord, alpha=1)
        return frame
    elif mode in sensor_list:
        frame = overlay_png(frame, sensor_icons[mode], sensor_icon_ul, alpha=1)
        frame = overlay_png(frame, back_button, back_button_ul, alpha=1)
        return frame


def mode_switch(mode):
    global click_x, click_y
    if click_x != -1 and click_y != -1:
        if mode == "selection":
            for sensor_name, bbox in sensor_icons_bbox_xyxy.items():
                if check_inbox(click_x, click_y, bbox):
                    mode = sensor_name
                    break
        elif mode in sensor_list:
            if check_inbox(click_x, click_y, back_button_bbox_xyxy):
                mode = "selection"

    click_x = -1
    click_y = -1
    return mode

    

# Video Capture
cap = cv2.VideoCapture(1)
cv2.namedWindow("frame")
click_x = -1
click_y = -1
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        global click_x, click_y
        click_x = x
        click_y = y
cv2.setMouseCallback("frame", mouse_callback)


mode = "selection"
while True:
    ret, frame = cap.read()
    if not ret:
        print("Cannot read frame")
        break
    # resize frame
    frame = cv2.resize(frame, frame_size)

    # Overlay everything
    frame = draw_on_frame(frame, mode)

    # mode switch
    mode = mode_switch(mode)

    # Display frame
    cv2.imshow("frame", frame)
    k = cv2.waitKey(1) & 0xff
    if k == 27:
        # exit if ESC pressed
        break