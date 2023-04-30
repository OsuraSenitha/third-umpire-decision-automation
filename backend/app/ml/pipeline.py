import numpy as np
from typing import Any
from .object_detector import ObjectDetectModel
from .yolo_image_segmentor import ImageSegmentModel
from .crease_cross_detector import CreaseCrossDetector
from .classifier import Classifier


class PipelineOutput:
    def __init__(
        self,
        batsman_box: np.ndarray,
        wicket_box: np.ndarray,
        batsman_result: bool,
        batsman_analysis_img_path: str,
        wicket_result: bool,
    ) -> None:
        self.annotations = [
            ["Batsman", *batsman_box.tolist()],
            ["Wicket", *wicket_box.tolist()],
        ]
        self.batsman_result = batsman_result
        self.batsman_analysis_img_path = batsman_analysis_img_path
        self.wicket_result = wicket_result


class Pipeline:
    def __init__(
        self, object_detect_model_path, image_segment_model_path, classifier_model_path
    ) -> None:
        object_detector = ObjectDetectModel(object_detect_model_path)
        image_segmentor = ImageSegmentModel(image_segment_model_path)
        crease_cross_detector = CreaseCrossDetector()
        classifier = Classifier(classifier_model_path)

        self.object_detector = object_detector
        self.image_segmentor = image_segmentor
        self.crease_cross_detector = crease_cross_detector
        self.classifier = classifier

    def __call__(self, img_path, batsman_analysis_image_path) -> Any:
        detection_output = self.object_detector(img_path)
        batsman_box = detection_output.getBoxFromLabel("Batsmen")
        wicket_box = detection_output.getBoxFromLabel("Wicket")

        batsman_seg = self.image_segmentor(img_path, batsman_box)
        batsman_result = self.crease_cross_detector(
            img_path, batsman_seg, wicket_box, batsman_analysis_image_path
        )
        wicket_result = self.classifier(img_path, wicket_box)

        results = PipelineOutput(
            batsman_box,
            wicket_box,
            batsman_result,
            batsman_analysis_image_path,
            wicket_result,
        )

        return results
