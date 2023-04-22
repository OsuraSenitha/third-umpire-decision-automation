import cv2 as cv
from .process import cvtAnnotationsTXT2LST
from typing import Tuple
import numpy as np

def drawRects(img, txt_cntnt, xywh=True, scaled=False, color=(0,255,0)):
    lines = txt_cntnt
    if type(lines) == str:
        lines = cvtAnnotationsTXT2LST(txt_cntnt)
    drawn_img = img.copy()
    H, W, _ = img.shape
    for line in lines:
        l, x, y, w, h = line
        xmin, ymin, xmax, ymax = x, y, w, h
        if not scaled and xywh:
            x, w = int(x*W), int(w*W)
            y, h = int(y*H), int(h*H)
        if not scaled and not xywh:
            xmin, ymin, xmax, ymax = int(x*W), int(y*H), int(w*W), int(h*H)
        if xywh:
            xmin, ymin = int(x-w/2), int(y-h/2)
            xmax, ymax = int(x+w/2),int(y+h/2)

        drawn_img = cv.rectangle(drawn_img, (xmin, ymin), (xmax, ymax), color, int(min(H, W)/100))

    return drawn_img

def drawSegment(
        img:np.ndarray, pts:Tuple[float], normalized:bool=True, color:Tuple[int]=[0,255,0], overlay_ratio:float=0.3
    ) -> np.ndarray:
    H, W, _ = img.shape
    X = np.array(pts[0::2])
    Y = np.array(pts[1::2])
    if normalized:
        X = (X*W).astype(int)
        Y = (Y*H).astype(int)
    
    conts = (np.stack((X, Y)).T)[np.newaxis, :]
    drawn_img = cv.drawContours(img.copy(), conts, -1, color)
    overlay_img = cv.fillPoly(img.copy(), pts=conts, color=color)
    drawn_img = (drawn_img*(1-overlay_ratio)+overlay_img*overlay_ratio).astype(drawn_img.dtype)

    point_radius = max(int(min(H, W)/160), 1)
    for x, y in zip(X, Y):
      drawn_img = cv.circle(drawn_img, center=(x, y), radius=point_radius, color=color, thickness=-1)

    return drawn_img