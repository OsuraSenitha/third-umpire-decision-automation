import colorsys
from typing import Tuple
import cv2 as cv

import numpy as np


def fillPts(
    img_shape: Tuple[int],
    pts_lst: Tuple,
    normalized: bool = True,
    color: Tuple[int] = [0, 255, 0],
) -> np.ndarray:
    H, W, _ = img_shape
    conts = []
    for pts in pts_lst:
        pts = np.array(pts)
        if normalized:
            pts[0::2] *= W
            pts[1::2] *= H
        pts = pts.astype(int).reshape((-1, 2))
        conts.append(pts)

    overlay = np.zeros(img_shape).astype(np.uint8)
    cv.fillPoly(overlay, pts=conts, color=color)

    return overlay


def drawSegment(
    img: np.ndarray,
    pts: Tuple[float],
    normalized: bool = True,
    color: Tuple[int] = [0, 255, 0],
    overlay_ratio: float = 0.3,
    line_width_scale: float = 0.01,
) -> np.ndarray:
    H, W, _ = img.shape
    X = np.array(pts[0::2])
    Y = np.array(pts[1::2])
    if normalized:
        X = (X * W).astype(int)
        Y = (Y * H).astype(int)

    conts = (np.stack((X, Y)).T)[np.newaxis, :]
    line_width = max(int(min(H, W) * line_width_scale), 1)
    drawn_img = cv.drawContours(img.copy(), conts, -1, color, line_width)
    # print(img.dtype, conts.dtype) # uint8 int64
    if overlay_ratio != 0:
        overlay_img = cv.fillPoly(img.copy(), pts=conts, color=color)
        drawn_img = (
            drawn_img * (1 - overlay_ratio) + overlay_img * overlay_ratio
        ).astype(drawn_img.dtype)

    # point_radius = max(int(line_width * 1.5), 1)
    # for x, y in zip(X, Y):
    #     drawn_img = cv.circle(
    #         drawn_img, center=(x, y), radius=point_radius, color=color, thickness=-1
    #     )

    return drawn_img


def drawSegments(
    img,
    seg_lines,
    normalized=True,
    color_overlay_ratio=0.3,
    has_class: bool = True,
    line_width_scale: float = 0.01,
    color: Tuple = None,
):
    drawn_img = img.copy()
    N = len(seg_lines)

    if color is None:
        c_hsv = [(x * 1.0 / N, 0.5, 0.5) for x in range(N)]
        c_rgb = np.array(list(map(lambda x: colorsys.hsv_to_rgb(*x), c_hsv)))
        c_rgb = (c_rgb * 255).astype(int).tolist()
    else:
        c_rgb = [color] * N

    # draw the dots and lines
    for obj, c in zip(seg_lines, c_rgb):
        if has_class:
            cls, *pts = obj
        else:
            pts = obj
        drawn_img = drawSegment(drawn_img, pts, normalized, c, 0, line_width_scale)

    overlays = np.zeros((N, *img.shape))
    for i, (obj, c) in enumerate(zip(seg_lines, c_rgb)):
        if has_class:
            cls, *pts = obj
        else:
            pts = obj
        overlay = fillPts(img.shape, [pts], normalized, c)
        overlays[i] = overlay

    overlay = (overlays.sum(axis=0) * color_overlay_ratio).astype(img.dtype)
    overlay_indices = overlay != 0
    overlayed_img = (
        drawn_img * (1 - color_overlay_ratio) + overlay * color_overlay_ratio
    ).astype(drawn_img.dtype)
    drawn_img[overlay_indices] = overlayed_img[overlay_indices]

    return drawn_img


def getx1y1x2y2FromRhoTheta(r_theta):
    arr = np.array(r_theta, dtype=np.float64)
    r, theta = arr
    a = np.cos(theta)
    b = np.sin(theta)
    x0, y0 = a * r, b * r
    x1, y1 = int(x0 + 1000 * (-b)), int(y0 + 1000 * (a))
    x2, y2 = int(x0 - 1000 * (-b)), int(y0 - 1000 * (a))
    return [x1, y1, x2, y2]


def drawLine(img, r_theta):
    x1, y1, x2, y2 = getx1y1x2y2FromRhoTheta(r_theta)
    cv.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
