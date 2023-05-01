import numpy as np
import cv2 as cv
from segment_anything import sam_model_registry, SamPredictor
from typing import Tuple, Union


def getBoundariesFromMask(
    mask, normalize=True, epsilon: float = None, threshold_area=100
) -> Tuple[Tuple[Union[int, float]]]:
    """
    Typical epsilon value: 1.3
    """
    H, W = mask.shape

    contours_raw, hierarchy_raw = cv.findContours(
        mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE
    )

    # reduce the number of points
    if epsilon is not None:
        contours_reduced = []
        for i in range(len(contours_raw)):
            reduced_contour = cv.approxPolyDP(contours_raw[i], epsilon, closed=True)
            contours_reduced.append(reduced_contour)
    else:
        contours_reduced = contours_raw

    if len(contours_reduced) > 0:
        # filter the contours by area
        contours, hierarchy = [], []
        for cnt, hir in zip(contours_reduced, hierarchy_raw[0]):
            area = cv.contourArea(cnt)
            if area > threshold_area:
                contours.append(cnt)
                hierarchy.append(hir)
        hierarchy = np.array(hierarchy)

        if len(contours) > 0:
            no_parent = hierarchy[:, 3] == -1
            no_parent_ids = np.where(no_parent)[0]
            contours = [contours[i].squeeze() for i in no_parent_ids]

            boundaries = []
            for cnt in contours:
                X, Y = cnt[:, 0], cnt[:, 1]

                if normalize:
                    X = X / W
                    Y = Y / H
                bnd = np.zeros(cnt.shape[0] * 2)
                bnd[0::2] = X
                bnd[1::2] = Y
                boundaries.append(bnd.tolist())

            return boundaries

    return []


class BatsmanSegmentor:
    def __init__(self, model_path, model_type="vit_h"):
        model_type = "vit_h"
        sam = sam_model_registry[model_type](checkpoint=model_path)
        predictor = SamPredictor(sam)

        self.predictor = predictor

    def _predict(self, bbox: np.ndarray) -> np.ndarray:
        masks, scores, logits = self.predictor.predict(
            point_coords=None,
            point_labels=None,
            box=(bbox),
            multimask_output=False,
        )

        return masks[0]

    def __call__(
        self, img_path: str, bbox: np.ndarray
    ) -> Tuple[Tuple[Union[int, float]]]:
        img = cv.imread(img_path)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.predictor.set_image(img)

        mask = self._predict(bbox)
        boundaries = getBoundariesFromMask(mask)

        return boundaries
