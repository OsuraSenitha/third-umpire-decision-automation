import os
from typing import Dict, Tuple
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from .visualize import getx1y1x2y2FromRhoTheta, drawLine, drawSegment, drawSegments


class CreaseCrossDetector:
    def _extract_focus_region(
        self, img, batsmen_segment, wicket_bbx=None, batsmen_possible_clearence=0.5
    ):
        if wicket_bbx is None:
            return img, batsmen_segment
        else:
            # output from the segmentation model (maped back to the image dimensios)
            H, W, _ = img.shape

            xmax = int(max(batsmen_segment[0::2]) * W)
            xmin = int(min(batsmen_segment[0::2]) * W)
            left_trim = xmin + int((xmax - xmin) * batsmen_possible_clearence)

            right_trim = int(wicket_bbx[0] * W)

            new_W = right_trim - left_trim

            new_x = np.array(batsmen_segment[0::2]) * W
            new_x -= left_trim
            new_x = new_x.clip(0) / new_W
            new_batsmen_segment = [*batsmen_segment]
            new_batsmen_segment[0::2] = new_x.tolist()

            focus_region = img[:, left_trim:right_trim, :]

            return focus_region, new_batsmen_segment

    def _get_line_intersection(
        self, line1: Tuple[float], line2: Tuple[float]
    ) -> Tuple[int]:
        """Finds the intersection of two lines given in Hesse normal form.

        Returns closest integer pixel locations.
        See https://stackoverflow.com/a/383527/5087436
        """
        rho1, theta1 = line1
        rho2, theta2 = line2
        A = np.array(
            [[np.cos(theta1), np.sin(theta1)], [np.cos(theta2), np.sin(theta2)]]
        )
        rank = np.linalg.matrix_rank(A)
        if rank == 1:
            return np.inf, np.inf
        else:
            b = np.array([[rho1], [rho2]])
            x0, y0 = np.linalg.solve(A, b)
            x0, y0 = int(np.round(x0)), int(np.round(y0))
            return x0, y0

    def _fill_pts(
        self,
        img_shape: Tuple[int],
        pts: Tuple,
        normalized: bool = True,
        color: Tuple[int] = [0, 255, 0],
    ) -> np.ndarray:
        H, W, _ = img_shape

        pts = np.array(pts)
        if normalized:
            pts[0::2] *= W
            pts[1::2] *= H
        pts = pts.astype(int).reshape((-1, 2))

        overlay = np.zeros(img_shape).astype(np.uint8)
        cv.fillPoly(overlay, pts=[pts], color=color)

        return overlay

    def _get_segments_intersections(self, img_shape, pts_1, pts_2):
        mask_1 = (
            self._fill_pts(
                img_shape=[*img_shape[:2], 1],
                pts=pts_1,
                color=1,
            )
            .squeeze()
            .astype(bool)
        )
        mask_2 = (
            self._fill_pts(
                img_shape=[*img_shape[:2], 1],
                pts=pts_2,
                color=1,
            )
            .squeeze()
            .astype(bool)
        )
        intersection = np.logical_and(mask_1, mask_2)

        return intersection

    def _find_crease_pass(
        self, focus_region, batsmen_segment, save_path="crease-pass-output.png"
    ):
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
            raise RuntimeError("Crease not found")
        first_line = lines[0][0]
        intersections = list(
            map(
                lambda r_theta: self._get_line_intersection(first_line, r_theta[0]),
                lines[1:],
            )
        )
        not_intersecting = list(
            map(
                lambda pt: pt[0] > W or pt[0] < 0 or pt[1] > H or pt[1] < 0,
                intersections,
            )
        )
        if True not in not_intersecting:
            raise RuntimeError("Crease not found")
        second_line_id = not_intersecting.index(True) + 1
        second_line = lines[second_line_id][0]

        # finding the intersection between the crease and the batsman
        first_line_xy = np.array(getx1y1x2y2FromRhoTheta(first_line)).astype(float)
        second_line_xy = np.array(getx1y1x2y2FromRhoTheta(second_line)).astype(float)
        first_line_xy[0::2] = first_line_xy[0::2] / W
        first_line_xy[1::2] = first_line_xy[1::2] / H
        second_line_xy[0::2] = second_line_xy[0::2] / W
        second_line_xy[1::2] = second_line_xy[1::2] / H
        crease_seg = [*np.roll(first_line_xy, 2).tolist(), *second_line_xy]
        intersection = self._get_segments_intersections(
            focus_region.shape, batsmen_segment, crease_seg
        )
        passed_crease = intersection.any()

        line_drawn_img = focus_region.copy()
        drawLine(line_drawn_img, first_line)
        drawLine(line_drawn_img, second_line)

        segs_drawn_img = focus_region.copy()
        segs_drawn_img = drawSegment(
            segs_drawn_img,
            batsmen_segment,
            color=[0, 255, 0],
            line_width_scale=0.004,
            overlay_ratio=0.3,
        )
        segs_drawn_img = drawSegment(
            segs_drawn_img,
            crease_seg,
            color=[0, 0, 255],
            line_width_scale=0.004,
            overlay_ratio=0.3,
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

        save_path = os.path.abspath(save_path)
        export_dir = os.path.split(save_path)[0]
        os.makedirs(export_dir, exist_ok=True)

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

        return passed_crease, save_path

    def __call__(
        self,
        img_path: str,
        batsman_seg: Tuple[Tuple],
        wicket_bbx: Tuple = None,
        save_path: str = "crease-pass-output.png",
    ) -> Dict:
        img = cv.imread(img_path)
        focus_region, batsmen_segment_focused = self._extract_focus_region(
            img, batsman_seg, wicket_bbx
        )
        passed, save_path = self._find_crease_pass(
            focus_region, batsmen_segment_focused, save_path
        )
        return passed
