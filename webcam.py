import cv2
from random import randint, choice
import numpy as np
import matplotlib.pyplot as plt
from PIL import ImageColor

def hexToBGR(hex):
    rgb = ImageColor.getcolor(hex, "RGB")
    return [rgb[2], rgb[1], rgb[0]]

colors = [hexToBGR("#ffb2a7"), hexToBGR("#e6739f"),
          hexToBGR("#cc0e74"), hexToBGR("#790c5a")]

def pickColor():
    return choice(colors)

def pickCircleSize():
    return randint(9, 20)


def moveCircles():
    for circle in trail:
        if circle is None:
            continue
        pos, _, _ = circle
        pos[1] -= 0.3


cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)
# this configures camera settings
if vc.isOpened():  # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

catpng = cv2.imread("cat.png", -1)
catpng = cv2.resize(catpng, (200, 220), interpolation=cv2.INTER_AREA)

trail = [None] * 150
alpha = 0.3
showcat = False


def overlaySmallOnBig(s_img, l_img, xoffset, yoffset):
    y1, y2 = yoffset - s_img.shape[0]/2, yoffset + s_img.shape[0]/2
    x1, x2 = xoffset -  s_img.shape[1]/2, xoffset + s_img.shape[1]/2

    if any(t < 0 or t> l_img.shape[0] for t in [x1, y1, x2, y2]):
        return

    alpha_s = s_img[:, :, 3] / 255.0
    alpha_s = alpha_s[:,:,None]
    alpha_l = 1.0 - alpha_s

    y1, y2, x1, x2  = int(y1), int(y2), int(x1), int(x2)

    l_img[y1:y2, x1:x2, :] = (alpha_s * s_img[:, :, :-1] + alpha_l * l_img[y1:y2, x1:x2, :])


def findCenterRed(img):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # lower mask (0-10)
    lower_red = np.array([0, 99, 99])
    upper_red = np.array([7, 255, 255])
    mask0 = cv2.inRange(img_hsv, lower_red, upper_red)

    # upper mask (170-180)
    lower_red = np.array([170, 99, 99])
    upper_red = np.array([180, 255, 255])
    mask1 = cv2.inRange(img_hsv, lower_red, upper_red)

    mask = mask0+mask1

    nonzero = np.argwhere(mask > 1)
    if (len(nonzero) <= 100):
        return None
    x, y = np.mean(nonzero, axis=0)
    print(x, y)
    return [y, x]


scaleWebcam = 2.0
ith = 0
currx, curry = 100, 100
width, height = 300, 100

showImg = True
while rval:
    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (int(
        frame.shape[1]*2), int(frame.shape[0]*2)), interpolation=cv2.INTER_CUBIC)
    copyForTransparency = frame.copy()
    print(ith)
    ith += 1
    # BGR
    movx, movy = 1*choice([-1, 1]), choice([-1, 1])
    currx += movx
    curry += movy
    center = findCenterRed(frame)
    if showImg:
        moveCircles()
        trail = trail[1:]
        if center is None:
            trail.append(None)
        else:
            trail.append((center, pickColor(), pickCircleSize()))

        for posAndColor in trail:
            if posAndColor is None:
                continue
            pos, color, size = posAndColor
            cv2.circle(copyForTransparency,
                       (int(pos[0]), int(pos[1])), size, color, thickness=-1)

    frame = cv2.addWeighted(frame, alpha, copyForTransparency, 1.0-alpha, 0)

    if showcat and center is not None:
        overlaySmallOnBig(catpng, frame, int(center[0]), int(center[1]))

    cv2.imshow("preview", frame)
    rval, frame = vc.read()
    key = cv2.waitKey(20)
    if key == 27:  # exit on ESC
        break
    if key == ord('t'):
        showImg = not showImg
    if key == ord('c'):
        trail = [None] * 150
    if key == ord('s'):
        showcat = not showcat

vc.release()
cv2.destroyWindow("preview")
