import cv2
import numpy as np

from utils import overlay_png, check_inbox



# Settings
frame_size = (1280, 720)
image_center = (int(frame_size[0] / 2), int(frame_size[1] / 2))
selection_ui_size = (600, 600)
icon_size = (90, 90)
sensor_list = ["vib1", "vib2", "mic1"]
sensor_icon_ul = (20, 20)
sensor_font_size = 1.5
back_button_size = (150, 100)
sensor_box_size = (135, 120)
sensor_box_ul = (sensor_icon_ul[0] - 11, sensor_icon_ul[1] - 15)
sensor_box_extended_size = (1000, 120)
sensor_box_extended_ul = sensor_box_ul
property_icon_size = (90, 90)
property_display_width = 858
property_display_beginx = sensor_box_ul[0] + 107
n_property = 4
icon_gap = (property_display_width - n_property * property_icon_size[0]) / (n_property + 1)
bar_property_icons_ul = [(int(property_display_beginx + icon_gap * (i + 1) + property_icon_size[0] * i), sensor_icon_ul[1]) for i in range(n_property)]
alpha_darken = 0.5



# Preparation
## sensor selection UI
sensor_selection = cv2.imread("./images/icons/selection.png", cv2.IMREAD_UNCHANGED)
sensor_selection = cv2.resize(sensor_selection, selection_ui_size)
## sensor icons
sensor_icons = {}
for sensor_name in sensor_list:
    icon = cv2.imread("./images/icons/" + sensor_name + ".png", cv2.IMREAD_UNCHANGED)
    icon = cv2.resize(icon, icon_size)
    sensor_icons[sensor_name] = icon
sensor_icons_bbox_xyxy = {"vib1": (394, 456, 495, 538),
                          "vib2": (785, 457, 885, 540),
                          "mic1": (610, 100, 668, 180)}
## mouse callback
click_x = -1
click_y = -1
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        global click_x, click_y
        click_x = x
        click_y = y
## back button
back_button = cv2.imread("./images/icons/backbutton.png", cv2.IMREAD_UNCHANGED)
back_button = cv2.resize(back_button, back_button_size)
back_button_ul = (frame_size[0] - back_button_size[0] - 20, 5)
back_button_bbox_xyxy = (back_button_ul[0], back_button_ul[1], back_button_ul[0] + back_button_size[0], back_button_ul[1] + back_button_size[1])
## sensor box
sensor_box = cv2.imread("./images/icons/sensor_box.png", cv2.IMREAD_UNCHANGED)
sensor_box = cv2.resize(sensor_box, sensor_box_size)
sensor_box_xyxy = (sensor_box_ul[0], sensor_box_ul[1], sensor_box_ul[0] + sensor_box_size[0], sensor_box_ul[1] + sensor_box_size[1])
sensor_box_extended = cv2.imread("./images/icons/sensor_box_extended.png", cv2.IMREAD_UNCHANGED)
sensor_box_extended = cv2.resize(sensor_box_extended, sensor_box_extended_size)
property_squeeze_box_xyxy = (sensor_box_ul[0] + 958, sensor_box_ul[1], sensor_box_ul[0] + sensor_box_extended_size[0], sensor_box_ul[1] + sensor_box_extended_size[1])
## bar property icons
bar_property_names = ["activities", "identification", "localization", "sound"]
bar_property_icons = {}
for property_name in bar_property_names:
    property_icon = cv2.imread("./images/icons/" + property_name + ".png", cv2.IMREAD_UNCHANGED)
    property_icon = cv2.resize(property_icon, property_icon_size)
    bar_property_icons[property_name] = property_icon
bar_property_icons_bbox_xyxy = {}
for i, property_name in enumerate(bar_property_names):
    bar_property_icons_bbox_xyxy[property_name] = (bar_property_icons_ul[i][0], bar_property_icons_ul[i][1],
                                                   bar_property_icons_ul[i][0] + property_icon_size[0], bar_property_icons_ul[i][1] + property_icon_size[1])
## property icons
property_names = ["walking", "running", "jumping", "dancing",
                  "localization1", "localization2", "localization3",
                  "identification1", "identification2", "identification3",
                  "sound1", "sound2", "sound3"]
property_icons = {}
for property_name in property_names:
    property_icon = cv2.imread("./images/icons/" + property_name + ".png", cv2.IMREAD_UNCHANGED)
    property_icon = cv2.resize(property_icon, property_icon_size)
    property_icons[property_name] = property_icon




# Video Capture
cap = cv2.VideoCapture(1)
cv2.namedWindow("frame")
cv2.setMouseCallback("frame", mouse_callback)
mode = "selection"

while True:
    ret, frame = cap.read()
    if not ret:
        print("Cannot read frame")
        break
    # resize frame
    frame = cv2.resize(frame, frame_size)

    # ovetlay UI
    if mode == "selection":
        # Overlay sensor selection UI
        coord = (image_center[0] - int(selection_ui_size[0] / 2),
                image_center[1] - int(selection_ui_size[1] / 2))
        frame = overlay_png(frame, sensor_selection, coord=coord, corner="ul")
    elif mode == "visualization":
        # Overlay sensor box
        frame = overlay_png(frame, sensor_box, coord=sensor_box_ul, corner="ul")

        # Overlay sensor icon
        coord = sensor_icon_ul
        frame = overlay_png(frame, sensor_icons[selected_sensor_name], coord=coord, corner="ul")
        # frame = cv2.putText(frame, selected_sensor_name,
        #                     (coord[0], coord[1] + icon_size[1] + 20),
        #                     cv2.FONT_HERSHEY_SIMPLEX, sensor_font_size, (0, 0, 0), 3, cv2.LINE_AA)

        
        # Overlay back button
        coord = back_button_ul
        frame = overlay_png(frame, back_button, coord=coord, corner="ul")
    elif mode == "property visualization":
        # Overlay sensor box
        frame = overlay_png(frame, sensor_box_extended, coord=sensor_box_extended_ul, corner="ul")

        # Overlay sensor icon
        coord = sensor_icon_ul
        frame = overlay_png(frame, sensor_icons[selected_sensor_name], coord=coord, corner="ul")
        # frame = cv2.putText(frame, selected_sensor_name,
        #                     (coord[0], coord[1] + icon_size[1] + 20),
        #                     cv2.FONT_HERSHEY_SIMPLEX, sensor_font_size, (0, 0, 0), 3, cv2.LINE_AA)

        # Overlay property icons on bar
        for i, property_name in enumerate(bar_property_names):
            coord = bar_property_icons_ul[i]
            icon = bar_property_icons[property_name]
            frame = overlay_png(frame, icon, coord=coord, corner="ul", alpha=alpha_darken)            
        
        # Overlay back button
        coord = back_button_ul
        frame = overlay_png(frame, back_button, coord=coord, corner="ul")






    # Show frame
    cv2.imshow("frame", frame)

    if click_x != -1 and click_y != -1:
        if mode == "selection":
            for sensor_name in sensor_list:
                if check_inbox(click_x, click_y, sensor_icons_bbox_xyxy[sensor_name]):
                    selected_sensor_name = sensor_name
                    mode = "visualization"
                    break
        elif mode == "visualization":
            if check_inbox(click_x, click_y, back_button_bbox_xyxy):
                mode = "selection"
            if check_inbox(click_x, click_y, sensor_box_xyxy):
                mode = "property visualization"
        elif mode == "property visualization":
            if check_inbox(click_x, click_y, back_button_bbox_xyxy):
                mode = "visualization"
            if check_inbox(click_x, click_y, property_squeeze_box_xyxy):
                mode = "visualization"

                





        click_x = -1
        click_y = -1
    
    k = cv2.waitKey(1) & 0xff
    if k == 27:
        # exit if ESC pressed
        break
    elif k == ord('q'):
        # back to selection mode
        mode = "selection"
    


