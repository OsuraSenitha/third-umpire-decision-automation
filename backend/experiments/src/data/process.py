import numpy as np
import cv2 as cv
from .analyze import img2ColorMat

def getBoundingBoxesFromSegmentation(seg_img, labels):
    colorMat = img2ColorMat(seg_img)
    box_groups = []
    for label in labels:
        color = label["color"]
        thresh = np.isin(colorMat, color).astype(np.uint8)
        contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        if len(contours)>0:
            no_parent = hierarchy[0][:,3] == -1
            bounding_rects = []
            # new_img = seg_img.copy()
            for cont, no_par in zip(contours, no_parent):
                if no_par:
                    box = cv.boundingRect(cont)
                    bounding_rects.append(box)
                    # x,y,w,h = box
                    # new_img = cv.rectangle(seg_img,(x,y),(x+w,y+h),(0,255,0),2)
            box_groups.append({"name": label["name"], "boxes": bounding_rects})

    return box_groups