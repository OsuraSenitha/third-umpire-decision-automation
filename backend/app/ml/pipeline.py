from typing import Dict
import numpy as np
from .crease_cross_detector import CreaseCrossDetector


class Pipeline:
    def __init__(self, detector, segmentor):
        self.detector = detector
        self.segmentor = segmentor
        self.crease_cross_detector = CreaseCrossDetector()

    def __call__(self, img_path: str) -> Dict:
        output = self.detector(img_path)

        if "Batsmen" in output.labels:
            wicket_bbx = None
            if "Wicket" in output.labels:
                wicket_box_id = output.labels.index("Batsmen")
                wicket_bbx = output.boxes[wicket_box_id]
            batsman_box_id = output.labels.index("Batsmen")
            box = output.boxes[batsman_box_id]

            boundaries = self.segmentor(img_path, np.array(box))
            crease_crossed_status, img_save_path = self.crease_cross_detector(
                img_path, boundaries, wicket_bbx, f"/tmp/crease-pass-output.png"
            )

            batsman_res = {
                "analysize_img_path": img_save_path,
                "crease_crossed_status": crease_crossed_status,
            }
        else:
            batsman_res = "No batsman detected"

        annotations = output.tolist()

        results = {"annotations": annotations, "batsman_res": batsman_res}

        return results
