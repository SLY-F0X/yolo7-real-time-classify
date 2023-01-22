import cv2
import numpy as np
import math
from PIL import Image
import pyautogui
import win32api
import win32con
import win32gui
import time
import sys
import keyboard

from pathlib import Path

import torch

from models.yolo import Model
from utils.general import check_requirements, set_logging
from utils.google_utils import attempt_download
from utils.torch_utils import select_device

#dependencies = ['torch', 'yaml']
#check_requirements(Path("D:/yolo7/").parent / 'D:/yolo7/requirements.txt', exclude=('pycocotools', 'thop'))
#set_logging()

def custom(path_or_model='D:/yolo7/', autoshape=True):
    model = torch.load(path_or_model, map_location=torch.device('cpu')) if isinstance(path_or_model, str) else path_or_model  # load checkpoint
    if isinstance(model, dict):
        model = model['ema' if model.get('ema') else 'model']  # load model

    hub_model = Model(model.yaml).to(next(model.parameters()).device)  # create
    hub_model.load_state_dict(model.float().state_dict())  # load state_dict
    hub_model.names = model.names  # class names
    if autoshape:
        hub_model = hub_model.autoshape()  # for file/URI/PIL/cv2/np inputs and NMS
    device = select_device('0' if torch.cuda.is_available() else 'cpu')  # default to GPU if available
    return hub_model.to(device)

model = custom(path_or_model='D:/yolo7/yolov7-tiny.pt')  # custom example
# model = create(name='yolov7', pretrained=True, channels=3, classes=80, autoshape=True)  # pretrained example

# Get the rect of the window

#hwnd = win32gui.FindWindow(None, 'Opera')
#rect = win32gui.GetWindowRect(hwnd)


while True:
    # Take a screenshot of the window and resize the image
    #scr = pyautogui.screenshot()
    scr = pyautogui.screenshot(region=(0, 0, 1024, 920))
    scr = scr.resize((640, 640))

# Convert the image to a numpy array
    img = np.array(scr)

#img = np.array(Image.open("test.png"))

    results = model(img)  # batched inference

#     x0,y0,x1,y1,confi,cla = results.xyxy[0][0].numpy()
#     print(x0,y0,x1,y1,confi,cla)
#     x0, y0, x1, y1, _, _ = results.xyxy[0][0].numpy().astype(int)
#     print(x0, y0, x1, y1, _, _)

    # Get the bounding boxes, confidence scores and class labels
    boxes = results.xyxy[0][:, :4].numpy().astype(int)
    confidences = results.xyxy[0][:, 4].numpy()
    class_labels = results.xyxy[0][:, 5].numpy()

    threshold = 0.5
    idxs = np.where((confidences > threshold) & (class_labels == 0) | (class_labels == 1))
    boxes = boxes[idxs]
    confidences = confidences[idxs]
    class_labels = class_labels[idxs]

# Apply non-maxima suppression
    nms_threshold = 0.6
    keep = cv2.dnn.NMSBoxes(boxes, confidences, threshold, nms_threshold)
    boxes = boxes[keep]
    class_labels = class_labels[keep]

# Draw bounding boxes on the image
    for box in boxes:
        x, y, w, h = box
        cv2.rectangle(img, (x, y), (x+w-50, y+h-50), (0, 255, 0), 1)

# Calculate distance of closest detection from center of the screen
    if len(boxes) > 0:
        min_dist = 9999
        min_at = 10
        for i in range(len(boxes)):
            (x, y, w, h) = boxes[i]
            label = f"{model.names[int(class_labels[i])]}: {confidences[i]:.2f}"
            cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            dist = math.sqrt(math.pow(img.shape[1]/2 - (x+w/2), 2) + math.pow(img.shape[0]/2 - (y+h/2), 2))
            if dist < min_dist:
                min_dist = dist
                min_at = i
        closest_box = boxes[min_at]
        x, y, w, h = closest_box

 # Get the coordinates of the center of the closest box
        center_x = closest_box[0] + closest_box[2] // 2
        center_y = closest_box[1] + closest_box[3] // 2
# Move the mouse cursor to the center of the closest box
        #pyautogui.moveTo(center_x, center_y)
        win32api.SetCursorPos((center_x, center_y))
# Simulate a click at the center of the closest detection
        #pyautogui.click(center_x, center_y)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, center_x, center_y, 0, 0)
        time.sleep(0.1)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, center_x, center_y, 0, 0)

    # WINDOW SCALLING
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))
    cv2.imshow("Detected Objects", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if keyboard.is_pressed('esc'):  # if key 'esc' is pressed
        cv2.destroyAllWindows()
        sys.exit()
        break
# Close all OpenCV windows
cv2.destroyAllWindows()
 
