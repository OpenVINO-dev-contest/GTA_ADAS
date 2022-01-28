from openvino.inference_engine import IECore
from heapq import _heapify_max
from operator import ne
import cv2, math, time, win32gui, win32api, win32con
import argparse
from win32gui import PumpMessages, PostQuitMessage
import numpy as np
from mss import mss
from ctypes import windll
import pygame

def parse_args() -> argparse.Namespace:
    """Parse and return command line arguments"""
    parser = argparse.ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    # fmt: off
    args.add_argument('-h', '--help', action='help', help='Show this help message and exit.')
    args.add_argument('-m', '--model', required=True, type=str,
                      help='Required. Path to an .xml or .onnx file with a trained model.')
    # fmt: on
    return parser.parse_args()
args = parse_args()

obj = {1:'vehicle', 2: 'pedestrian'}
pygame.init()
MoniterWidth = 1920
MoniterHeight = 1080
AimWidth = 672
AimHeight = 384
x = 0
y = 0
screen = pygame.display.set_mode((1920, 1080), pygame.HWSURFACE)
fuchsia = (255, 0, 128)
red = (255,0,0)
green = (0,255,0)
blue = (0,0,255)
NOSIZE = 1
NOMOVE = 2
TOPMOST = -1
NOT_TOPMOST = -2
hwnd = pygame.display.get_wm_info()["window"]
win32gui.SetWindowLong(hwnd, win32con.GWL_EXSTYLE,
                       win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE) | win32con.WS_EX_LAYERED)
win32gui.SetLayeredWindowAttributes(hwnd, win32api.RGB(*fuchsia), 0, win32con.LWA_COLORKEY)
SetWindowPos = windll.user32.SetWindowPos

def alwaysOnTop(yesOrNo):
    zorder = (NOT_TOPMOST, TOPMOST)[yesOrNo]
    hwnd = pygame.display.get_wm_info()['window']
    SetWindowPos(hwnd, zorder, 0, 0, 0, 0, NOMOVE|NOSIZE)
alwaysOnTop(True)


def drawText(text, x, y, backgroundColor, textColor, textSize):
    font = pygame.font.Font('freesansbold.ttf', textSize) 
    fontText = font.render(text, True, textColor, backgroundColor) 
    textRect = fontText.get_rect()  
    textRect.center = (x, y) 
    screen.blit(fontText, textRect)


monitor = {"top": int(MoniterHeight/2-AimHeight/2), "left": int(MoniterWidth/2-AimWidth/2), "width": AimWidth, "height": AimHeight}
sct = mss()

# ---------------------------Initialize inference engine core--------------------------------------------------
ie = IECore()

 # ---------------------------Read a model in OpenVINO Intermediate Representation or ONNX format---------------
net = ie.read_network(model=args.model)

# ---------------------------Configure input & output----------------------------------------------------------
input_blob = next(iter(net.input_info))
output_blob = next(iter(net.outputs))

# ---------------------------Loading model to the device-------------------------------------------------------
exec_net = ie.load_network(network=net, device_name="CPU", num_requests=2)
frame = sct.grab(monitor)
curr_id=0
next_id=1
while True:
    # ---------------------------Prepare input---------------------------------------------------------------------
    next_frame= sct.grab(monitor)
    screen.fill(fuchsia)
    pygame.draw.rect(screen, blue, [MoniterWidth/2-AimWidth/2, MoniterHeight/2-AimHeight/2, AimWidth, AimHeight], 1)
    timer = cv2.getTickCount()
    frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGBA2BGR)
    frame = frame.transpose((2, 0, 1))
    frame = np.expand_dims(frame, axis=0)
    
    # ---------------------------Do inference----------------------------------------------------------------------
    exec_net.requests[curr_id].async_infer(inputs={input_blob: frame})
    if exec_net.requests[curr_id].wait(-1)==0:
        
        # ---------------------------Process output--------------------------------------------------------------------
        res = exec_net.requests[curr_id].output_blobs[output_blob].buffer
        detections = res.reshape(-1, 7)
        for i, detection in enumerate(detections):
            image_id, label, confidence, xmin, ymin, xmax, ymax = detection
            if confidence > 0.5:
                print(xmin)
                xmin = int(xmin*672)
                ymin = int(ymin*384)
                xmax = int(xmax*672)
                ymax = int(ymax*384)
                w=int(xmax-xmin)
                h=int(ymax-ymin)
                if ((AimHeight-(ymin+h))<20):
                    pygame.draw.rect(screen, red, [xmin + (MoniterWidth/2 - AimWidth/2), ymin + (MoniterHeight/2 - AimHeight/2), w, h], 15)
                    drawText("Caution!",xmin + (MoniterWidth/2 - AimWidth/2)+20 ,ymin + (MoniterHeight/2 - AimHeight/2)-5, backgroundColor=fuchsia, textColor=red, textSize=60)
                else:
                    pygame.draw.rect(screen, green, [xmin + (MoniterWidth/2 - AimWidth/2), ymin + (MoniterHeight/2 - AimHeight/2)-5, w, h], 2)
                    drawText(obj[label],xmin + (MoniterWidth/2 - AimWidth/2)+10 ,ymin + (MoniterHeight/2 - AimHeight/2)-5, backgroundColor=fuchsia, textColor=green, textSize=16)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
    drawText('Detection FPS: ' + str(int(fps)),150,25, backgroundColor=fuchsia, textColor=green, textSize=32)
    pygame.display.update()
    cv2.waitKey(1)
    frame=next_frame
    curr_id, next_id = next_id, curr_id
