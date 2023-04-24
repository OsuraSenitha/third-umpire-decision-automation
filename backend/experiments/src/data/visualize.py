import cv2 as cv
from .process import cvtAnnotationsTXT2LST
from typing import Tuple
import numpy as np
import colorsys
import matplotlib.pyplot as plt
from .analyze import getLineIntersection


def drawRects(
    img,
    txt_cntnt,
    xywh=True,
    scaled=False,
    color=(0, 255, 0),
    line_width_scale: float = 0.01,
):
    lines = txt_cntnt
    if type(lines) == str:
        lines = cvtAnnotationsTXT2LST(txt_cntnt)
    drawn_img = img.copy()
    H, W, _ = img.shape
    for line in lines:
        l, x, y, w, h = line
        xmin, ymin, xmax, ymax = x, y, w, h
        if not scaled and xywh:
            x, w = int(x * W), int(w * W)
            y, h = int(y * H), int(h * H)
        if not scaled and not xywh:
            xmin, ymin, xmax, ymax = int(x * W), int(y * H), int(w * W), int(h * H)
        if xywh:
            xmin, ymin = int(x - w / 2), int(y - h / 2)
            xmax, ymax = int(x + w / 2), int(y + h / 2)

        line_width = max(int(min(H, W) * line_width_scale), 1)
        drawn_img = cv.rectangle(
            drawn_img, (xmin, ymin), (xmax, ymax), color, line_width
        )

    return drawn_img


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

    point_radius = max(int(min(H, W) / 160), 1)
    for x, y in zip(X, Y):
        drawn_img = cv.circle(
            drawn_img, center=(x, y), radius=point_radius, color=color, thickness=-1
        )

    return drawn_img


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


def drawSegments(
    img,
    pts_obj,
    normalized=True,
    color_overlay_ratio=0.3,
    has_class: bool = True,
    line_width_scale: float = 0.01,
    color: Tuple = None,
):
    if type(pts_obj) == str:
        seg_lines = cvtAnnotationsTXT2LST(pts_obj)
    else:
        seg_lines = pts_obj
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


def getSegmentsIntersections(img_shape, segs_1, segs_2):
    pts_1 = list(map(lambda pt_line: pt_line[1:], segs_1))
    pts_2 = list(map(lambda pt_line: pt_line[1:], segs_2))
    mask_1 = (
        fillPts(
            img_shape=[*img_shape[:2], 1],
            pts_lst=pts_1,
            color=1,
        )
        .squeeze()
        .astype(bool)
    )
    mask_2 = (
        fillPts(
            img_shape=[*img_shape[:2], 1],
            pts_lst=pts_2,
            color=1,
        )
        .squeeze()
        .astype(bool)
    )
    intersection = np.logical_and(mask_1, mask_2)

    return intersection


def findCreasePass(focus_region, batsmen_segments, visualize=False):
    H, W, _ = focus_region.shape

    gray = cv.cvtColor(focus_region, cv.COLOR_BGR2GRAY)
    t_lower = 50  # Lower Threshold
    t_upper = 150  # Upper threshold

    # Applying the Canny Edge filter
    # TODO: make the appature size dependant on the image dimensions
    edges = cv.Canny(gray, t_lower, t_upper)

    # Find the two bounding lines of the crease
    threshold = 100
    min_theta = -30 / 180 * np.pi
    max_theta = 30 / 180 * np.pi
    lines = cv.HoughLines(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=threshold,
        max_theta=max_theta,
        min_theta=min_theta,
    )
    if len(lines) == 0:
        raise Exception("Crease not found")
    first_line = lines[0][0]
    intersections = list(
        map(lambda r_theta: getLineIntersection(first_line, r_theta[0]), lines[1:])
    )
    not_intersecting = list(
        map(lambda pt: pt[0] > W or pt[0] < 0 or pt[1] > H or pt[1] < 0, intersections)
    )
    if True not in not_intersecting:
        raise Exception("Crease not found")
    second_line_id = not_intersecting.index(True) + 1
    second_line = lines[second_line_id][0]

    # finding the intersection between the crease and the batsman
    first_line_xy = np.array(getx1y1x2y2FromRhoTheta(first_line)).astype(float)
    second_line_xy = np.array(getx1y1x2y2FromRhoTheta(second_line)).astype(float)
    first_line_xy[0::2] = first_line_xy[0::2] / W
    first_line_xy[1::2] = first_line_xy[1::2] / H
    second_line_xy[0::2] = second_line_xy[0::2] / W
    second_line_xy[1::2] = second_line_xy[1::2] / H
    crease_segs = [[0, *np.roll(first_line_xy, 2), *second_line_xy]]
    intersection = getSegmentsIntersections(
        focus_region.shape, batsmen_segments, crease_segs
    )
    passed_crease = intersection.any()

    if visualize:
        line_drawn_img = focus_region.copy()
        drawLine(line_drawn_img, first_line)
        drawLine(line_drawn_img, second_line)

        segs_drawn_img = drawSegments(
            focus_region,
            batsmen_segments,
            color=[0, 255, 0],
            line_width_scale=0.004,
            color_overlay_ratio=1,
        )
        segs_drawn_img = drawSegments(
            segs_drawn_img,
            crease_segs,
            color=[0, 0, 255],
            line_width_scale=0.004,
            color_overlay_ratio=1,
        )

        intersection_img = focus_region.copy()
        intersection_img[intersection] = [0, 0, 255]

        fig, ax = plt.subplots(2, 3, figsize=(9, 6))
        ax[0][0].imshow(cv.cvtColor(focus_region, cv.COLOR_BGR2RGB))
        ax[0][0].set_title("Focused region")
        ax[0][1].imshow(gray, cmap="gray")
        ax[0][1].set_title("Gray image")
        ax[0][2].imshow(edges, cmap="gray")
        ax[0][2].set_title("Detected edges")
        ax[1][0].imshow(cv.cvtColor(line_drawn_img, cv.COLOR_BGR2RGB))
        ax[1][0].set_title("Detected lines")
        ax[1][1].imshow(cv.cvtColor(segs_drawn_img, cv.COLOR_BGR2RGB))
        ax[1][1].set_title("Batsman and crease contours")
        ax[1][2].imshow(cv.cvtColor(intersection_img, cv.COLOR_BGR2RGB))
        ax[1][2].set_title("Contour intersection")

        for i in range(2):
            for j in range(3):
                ax[i][j].set_xticks([])
                ax[i][j].set_yticks([])

        plt.show()

    return passed_crease
