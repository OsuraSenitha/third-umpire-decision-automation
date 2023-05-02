import numpy as np
import cv2 as cv
import json
from typing import Any
from .object_detector import ObjectDetectModel
from .yolo_image_segmentor import ImageSegmentModel
from .crease_cross_detector import CreaseCrossDetector
from .wicket_classifier import WicketClassifier


class PipelineOutput:
    def __init__(
        self,
        batsman_box: np.ndarray,
        wicket_box: np.ndarray,
        batsman_result: bool,
        batsman_analysis_img_path: str,
        wicket_result: bool,
        wicket_img_path: str,
        img_path: str,
    ) -> None:
        img = cv.imread(img_path)
        H, W, _ = img.shape
        batsman_box[0::2] *= W
        batsman_box[1::2] *= H
        wicket_box[0::2] *= W
        wicket_box[1::2] *= H
        batsman_box = batsman_box.astype(int)
        wicket_box = wicket_box.astype(int)

        self.annotations = [
            ["Batsman", *batsman_box.tolist()],
            ["Wicket", *wicket_box.tolist()],
        ]
        self.batsman_result = batsman_result
        self.batsman_analysis_img_path = batsman_analysis_img_path
        self.wicket_result = wicket_result
        self.wicket_img_path = wicket_img_path

    def __str__(self) -> str:
        self_desc = {
            "annotations": self.annotations,
            "batsman_result": str(self.batsman_result),
            "batsman_analysis_img_path": self.batsman_analysis_img_path,
            "wicket_result": str(self.wicket_result),
            "wicket_img_path": self.wicket_img_path,
        }
        self_desc = json.dumps(self_desc, indent=2)
        return self_desc


class Pipeline:
    def __init__(
        self, object_detect_model_path, image_segment_model_path, classifier_model_path
    ) -> None:
        object_detector = ObjectDetectModel(object_detect_model_path)
        image_segmentor = ImageSegmentModel(image_segment_model_path)
        crease_cross_detector = CreaseCrossDetector()
        classifier = WicketClassifier(classifier_model_path)

        self.object_detector = object_detector
        self.image_segmentor = image_segmentor
        self.crease_cross_detector = crease_cross_detector
        self.classifier = classifier

    def __call__(self, img_path, batsman_analysis_image_path, wicket_img_path) -> Any:
        detection_output = self.object_detector(img_path)
        batsman_box = detection_output.getBoxFromLabel("Batsmen")
        wicket_box = detection_output.getBoxFromLabel("Wicket")

        try:
            batsman_seg = self.image_segmentor(img_path, batsman_box)
            batsman_result = self.crease_cross_detector(
                img_path, batsman_seg, wicket_box, batsman_analysis_image_path
            )
        except RuntimeError as e:
            if str(e) == "Batsman segmentation was not found":
                print("ALGO LOGS: Batsman segmentation was not found")
                batsman_result = None
                batsman_analysis_image_path = None
            else:
                raise e

        wicket_result = self.classifier(img_path, wicket_box)
        self._trim_wicket(img_path, wicket_box, wicket_img_path)

        results = PipelineOutput(
            batsman_box,
            wicket_box,
            batsman_result,
            batsman_analysis_image_path,
            wicket_result,
            wicket_img_path,
            img_path,
        )

        return results

    def _trim_wicket(self, img_path, wicket_box, wicket_img_path):
        img = cv.imread(img_path)
        H, W, _ = img.shape
        wicket_box = wicket_box.copy()
        wicket_box[0::2] *= W
        wicket_box[1::2] *= H
        x1, y1, x2, y2 = wicket_box.astype(int)
        wicket_img = img[y1:y2, x1:x2, :]
        cv.imwrite(wicket_img_path, wicket_img)
