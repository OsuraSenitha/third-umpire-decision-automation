import cv2 as cv
from .process import cvtAnnotationsTXT2LST
from typing import Tuple
import numpy as np
import colorsys

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

def drawSegment(img:np.ndarray, pts:Tuple[float], normalized:bool=True, color:Tuple[int]=[0,255,0], overlay_ratio:float=0.3) -> np.ndarray:
    H, W, _ = img.shape
    X = np.array(pts[0::2])
    Y = np.array(pts[1::2])
    if normalized:
        X = (X*W).astype(int)
        Y = (Y*H).astype(int)
    
    conts = (np.stack((X, Y)).T)[np.newaxis, :]
    drawn_img = cv.drawContours(img.copy(), conts, -1, color)

    if overlay_ratio != 0:
        overlay_img = cv.fillPoly(img.copy(), pts=conts, color=color)
        drawn_img = (drawn_img*(1-overlay_ratio)+overlay_img*overlay_ratio).astype(drawn_img.dtype)

    point_radius = max(int(min(H, W)/160), 1)
    for x, y in zip(X, Y):
      drawn_img = cv.circle(drawn_img, center=(x, y), radius=point_radius, color=color, thickness=-1)

    return drawn_img

def getOverlay(img, pts, normalized, color):
    H, W, _ = img.shape
    X = np.array(pts[0::2])
    Y = np.array(pts[1::2])
    if normalized:
        X = (X*W).astype(int)
        Y = (Y*H).astype(int)

    conts = (np.stack((X, Y)).T)[np.newaxis, :]
    overlay = np.zeros(img.shape).astype(np.uint8)
    
    overlay = cv.fillPoly(overlay, pts=conts, color=color)

    return overlay


def drawSegments(img, pts_obj, normalized=True, color_overlay_ratio=0.3):
    if type(pts_obj) == str:
        pts_lines = cvtAnnotationsTXT2LST(pts_obj)
    else:
        pts_lines = pts_obj
    drawn_img = img.copy()
    N = len(pts_lines)
    c_hsv = [(x*1.0/N, 0.5, 0.5) for x in range(N)]
    c_rgb = np.array(list(map(lambda x: colorsys.hsv_to_rgb(*x), c_hsv)))
    c_rgb = (c_rgb*255).astype(int).tolist()

    # draw the dots and lines
    for obj, c in zip(pts_lines, c_rgb):
        cls, *pts = obj
        drawn_img = drawSegment(drawn_img, pts, normalized, c, 0)

    overlays = np.zeros((N, *img.shape))
    for i, (obj, c) in enumerate(zip(pts_lines, c_rgb)):
        cls, *pts = obj
        overlay = getOverlay(img, pts, normalized, c)
        overlays[i] = overlay

    overlay = (overlays.sum(axis=0)*color_overlay_ratio).astype(img.dtype)
    overlay_indices = overlay!=0
    overlayed_img = (drawn_img*(1-color_overlay_ratio)+overlay*color_overlay_ratio).astype(drawn_img.dtype)
    drawn_img[overlay_indices] = overlayed_img[overlay_indices]

    return drawn_img