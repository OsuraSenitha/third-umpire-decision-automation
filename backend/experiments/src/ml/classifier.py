from typing import Any
import numpy as np


class Classifier:
    def __init__(self, model_path: str) -> None:
        self.model_path = model_path

    def __call__(self, img_path: str, bounding_box: np.ndarray) -> bool:
        pred = bool(np.random.randint(0, 2))
        return pred
