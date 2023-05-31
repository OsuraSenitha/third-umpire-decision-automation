from typing import Dict
import numpy as np
import cv2 as cv
import onnxruntime as ort


class WicketClassifier:
    def __init__(self, model_path: str) -> None:
        session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        model_inputs = session.get_inputs()
        input_names = [ip.name for ip in model_inputs]
        model_outputs = session.get_outputs()
        output_names = [op.name for op in model_outputs]
        input_width, input_height = model_inputs[0].shape[2:]

        self.session = session
        self.input_names = input_names
        self.input_width = input_width
        self.input_height = input_height
        self.output_names = output_names

    def _prepare_input(self, img: np.ndarray) -> Dict[str, np.ndarray]:
        # prepare input
        input_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        # Resize input image
        input_img = cv.resize(input_img, (self.input_width, self.input_height))
        # Scale input pixel values to 0 to 1
        input_img = input_img / 255.0
        input_img = input_img.transpose(2, 0, 1)
        input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)
        input = {self.input_names[0]: input_tensor}
        return input

    def _process_output(self, output):
        output = np.bool_(output[0][0].argmax())
        return output

    def _crop_wicket(self, img, wicket_bbx):
        H, W, _ = img.shape
        wicket_bbx = wicket_bbx.copy()
        wicket_bbx[0::2] *= W
        wicket_bbx[1::2] *= H
        wicket_bbx = wicket_bbx.astype(int)
        x1, y1, x2, y2 = wicket_bbx
        max_dim = max(y2 - y1, x2 - x1)
        pad_ratio = 0.2
        new_size = int(max_dim * (1 + pad_ratio))
        x1_n = int(x1 - (new_size - (x2 - x1)) / 2)
        x1_n = max(0, x1_n)
        x2_n = x1_n + new_size
        x2_n = min(x2_n, W)
        y1_n = int(y1 - (new_size - (y2 - y1)) / 2)
        y1_n = max(0, y1_n)
        y2_n = y1_n + new_size
        y2_n = min(y2_n, W)
        img_croped = img[y1_n:y2_n, x1_n:x2_n, :]

        return img_croped

    def __call__(self, img_path: str, bounding_box: np.ndarray) -> bool:
        img = cv.imread(img_path)
        wicket_img = self._crop_wicket(img, bounding_box)
        input = self._prepare_input(wicket_img)
        output = self.session.run(self.output_names, input)
        # [0.39817, 0.823301]
        pred = self._process_output(output)

        return pred
