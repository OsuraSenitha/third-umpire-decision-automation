from typing import Any
from .object_detector import ObjectDetectModel
from .yolo_image_segmentor import ImageSegmentModel
from .crease_cross_detector import CreaseCrossDetector


class Pipeline:
    def __init__(self, object_detect_model_path, image_segment_model_path) -> None:
        object_detector = ObjectDetectModel(object_detect_model_path)
        image_segmentor = ImageSegmentModel(image_segment_model_path)
        crease_cross_detector = CreaseCrossDetector()

        self.object_detector = object_detector
        self.image_segmentor = image_segmentor
        self.crease_cross_detector = crease_cross_detector

    def __call__(self, img_path, batsman_analysis_image_path) -> Any:
        detection_output = self.object_detector(img_path)
        batsman_box = detection_output.getBoxFromLabel("Batsmen")
        wicket_bbx = detection_output.getBoxFromLabel("Wicket")

        batsman_seg = self.image_segmentor(img_path, batsman_box)
        batsman_crossed = self.crease_cross_detector(
            img_path, batsman_seg, wicket_bbx, batsman_analysis_image_path
        )

        batsman_result = {
            "batsman_crossed": batsman_crossed,
        }

        results = {"batsman_result": batsman_result}

        return results
