import cv2 as cv
from .process import cvtAnnotationsTXT2LST

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