import cv2

from utils import overlay_png


# Settings
frame_size = (1280, 720)
image_center = (int(frame_size[0] / 2), int(frame_size[1] / 2))
selection_ui_size = (600, 600)
selection_ui_coord = (image_center[0] - int(selection_ui_size[0] / 2),
                      image_center[1] - int(selection_ui_size[1] / 2))



# Preparation
## sensor selection UI
sensor_selection = cv2.imread("./images/icons/selection.png", cv2.IMREAD_UNCHANGED)
sensor_selection = cv2.resize(sensor_selection, selection_ui_size)





def draw_on_frame(frame, mode, click_x, click_y):
    if mode == "selection":
        frame = overlay_png(frame, sensor_selection, selection_ui_coord, alpha=1)
        return frame
    

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
    frame = draw_on_frame(frame, mode, click_x, click_y)

    # Display frame
    cv2.imshow("frame", frame)
    k = cv2.waitKey(1) & 0xff
    if k == 27:
        # exit if ESC pressed
        break