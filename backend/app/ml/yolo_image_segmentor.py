import math
import time
import cv2 as cv

import numpy as np
import onnxruntime

from typing import Tuple, Union

class_names = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


def nms(boxes, scores, iou_threshold):
    # Sort by score
    sorted_indices = np.argsort(scores)[::-1]

    keep_boxes = []
    while sorted_indices.size > 0:
        # Pick the last box
        box_id = sorted_indices[0]
        keep_boxes.append(box_id)

        # Compute IoU of the picked box with the rest
        ious = compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])

        # Remove boxes with IoU over the threshold
        keep_indices = np.where(ious < iou_threshold)[0]

        sorted_indices = sorted_indices[keep_indices + 1]

    return keep_boxes


def compute_iou(box, boxes):
    # Compute xmin, ymin, xmax, ymax for both boxes
    xmin = np.maximum(box[0], boxes[:, 0])
    ymin = np.maximum(box[1], boxes[:, 1])
    xmax = np.minimum(box[2], boxes[:, 2])
    ymax = np.minimum(box[3], boxes[:, 3])

    # Compute intersection area
    intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)

    # Compute union area
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = box_area + boxes_area - intersection_area

    # Compute IoU
    iou = intersection_area / union_area

    return iou


def xywh2xyxy(x):
    # Convert bounding box (x, y, w, h) to bounding box (x1, y1, x2, y2)
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


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


class ImageSegmentModel:
    def __init__(self, path, conf_thres=0.7, iou_thres=0.5, num_masks=32):
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres
        self.num_masks = num_masks

        # Initialize model
        self.initialize_model(path)

    def _get_batsman_boundary(
        self, bounding_box, image_shape, mask_maps
    ) -> Tuple[float]:
        x1, y1, x2, y2 = bounding_box
        resized_mask_shape = [mask_maps.shape[0], *image_shape[:2]]
        mask_maps_resized = np.zeros(resized_mask_shape).astype(bool)
        for i, mask_map in enumerate(mask_maps):
            mask_maps_resized[i, y1:y2, x1:x2] = mask_map

        batsmen_boundaries = getBoundariesFromMask(
            mask_maps_resized[0].astype(np.uint8)
        )

        # select the boundary with the maximum area
        max_area = 0
        max_bnd = None
        for bnd in batsmen_boundaries:
            cnt = np.array(bnd).reshape((-1, 1, 2)).astype(np.float32)
            area = cv.contourArea(cnt)
            if area > max_area:
                max_area = area
                max_bnd = bnd

        return max_bnd

    def _load_img(self, image_path, bounding_box):
        image = cv.imread(image_path)
        image_shape = image.shape
        bounding_box_scaled = bounding_box.copy()
        bounding_box_scaled[0::2] = bounding_box_scaled[0::2] * image_shape[1]
        bounding_box_scaled[1::2] = bounding_box_scaled[1::2] * image_shape[0]
        bounding_box_scaled = bounding_box_scaled.astype(int)

        x1, y1, x2, y2 = bounding_box_scaled
        image_trimmed = image[y1:y2, x1:x2, :]
        return image_trimmed, image_shape, bounding_box_scaled

    def __call__(self, image_path, bounding_box) -> Tuple[float]:
        # TODO: make the boundary square shaped

        image_trimmed, image_shape, bounding_box_scaled = self._load_img(
            image_path, bounding_box
        )
        boxes, scores, class_ids, mask_maps = self.segment_objects(image_trimmed)
        if mask_maps is None:
            raise RuntimeError("Batsman segmentation was not found")
        boundaries = self._get_batsman_boundary(
            bounding_box_scaled, image_shape, mask_maps
        )

        return boundaries

    def initialize_model(self, path):
        self.session = onnxruntime.InferenceSession(
            path, providers=["CPUExecutionProvider"]
        )
        # Get model info
        self.get_input_details()
        self.get_output_details()

    def segment_objects(self, image):
        input_tensor = self.prepare_input(image)

        # Perform inference on the image
        outputs = self.inference(input_tensor)

        self.boxes, self.scores, self.class_ids, mask_pred = self.process_box_output(
            outputs[0]
        )
        self.mask_maps = self.process_mask_output(mask_pred, outputs[1])

        return self.boxes, self.scores, self.class_ids, self.mask_maps

    def prepare_input(self, image):
        self.img_height, self.img_width = image.shape[:2]

        input_img = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        # Resize input image
        input_img = cv.resize(input_img, (self.input_width, self.input_height))

        # Scale input pixel values to 0 to 1
        input_img = input_img / 255.0
        input_img = input_img.transpose(2, 0, 1)
        input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)

        return input_tensor

    def inference(self, input_tensor):
        start = time.perf_counter()
        outputs = self.session.run(
            self.output_names, {self.input_names[0]: input_tensor}
        )

        # print(f"Inference time: {(time.perf_counter() - start)*1000:.2f} ms")
        return outputs

    def process_box_output(self, box_output):
        predictions = np.squeeze(box_output).T
        num_classes = box_output.shape[1] - self.num_masks - 4

        # Filter out object confidence scores below threshold
        scores = np.max(predictions[:, 4 : 4 + num_classes], axis=1)
        predictions = predictions[scores > self.conf_threshold, :]
        scores = scores[scores > self.conf_threshold]

        if len(scores) == 0:
            return [], [], [], np.array([])

        box_predictions = predictions[..., : num_classes + 4]
        mask_predictions = predictions[..., num_classes + 4 :]

        # Get the class with the highest confidence
        class_ids = np.argmax(box_predictions[:, 4:], axis=1)

        # Get bounding boxes for each object
        boxes = self.extract_boxes(box_predictions)

        # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
        indices = nms(boxes, scores, self.iou_threshold)

        return (
            boxes[indices],
            scores[indices],
            class_ids[indices],
            mask_predictions[indices],
        )

    def process_mask_output(self, mask_predictions, mask_output):
        if mask_predictions.shape[0] == 0:
            return None

        mask_output = np.squeeze(mask_output)

        # Calculate the mask maps for each box
        num_mask, mask_height, mask_width = mask_output.shape  # CHW
        masks = sigmoid(mask_predictions @ mask_output.reshape((num_mask, -1)))
        masks = masks.reshape((-1, mask_height, mask_width))

        # Downscale the boxes to match the mask size
        scale_boxes = self.rescale_boxes(
            self.boxes, (self.img_height, self.img_width), (mask_height, mask_width)
        )

        # For every box/mask pair, get the mask map
        mask_maps = np.zeros((len(scale_boxes), self.img_height, self.img_width))
        blur_size = (
            max(int(self.img_width / mask_width), 1),
            max(int(self.img_height / mask_height), 1),
        )
        for i in range(len(scale_boxes)):
            scale_x1 = int(math.floor(scale_boxes[i][0]))
            scale_y1 = int(math.floor(scale_boxes[i][1]))
            scale_x2 = int(math.ceil(scale_boxes[i][2]))
            scale_y2 = int(math.ceil(scale_boxes[i][3]))

            x1 = int(math.floor(self.boxes[i][0]))
            y1 = int(math.floor(self.boxes[i][1]))
            x2 = int(math.ceil(self.boxes[i][2]))
            y2 = int(math.ceil(self.boxes[i][3]))

            scale_crop_mask = masks[i][scale_y1:scale_y2, scale_x1:scale_x2]
            crop_mask = cv.resize(
                scale_crop_mask, (x2 - x1, y2 - y1), interpolation=cv.INTER_CUBIC
            )

            crop_mask = cv.blur(crop_mask, blur_size)

            crop_mask = (crop_mask > 0.5).astype(np.uint8)
            mask_maps[i, y1:y2, x1:x2] = crop_mask

        return mask_maps

    def extract_boxes(self, box_predictions):
        # Extract boxes from predictions
        boxes = box_predictions[:, :4]

        # Scale boxes to original image dimensions
        boxes = self.rescale_boxes(
            boxes,
            (self.input_height, self.input_width),
            (self.img_height, self.img_width),
        )

        # Convert boxes to xyxy format
        boxes = xywh2xyxy(boxes)

        # Check the boxes are within the image
        boxes[:, 0] = np.clip(boxes[:, 0], 0, self.img_width)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, self.img_height)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, self.img_width)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, self.img_height)

        return boxes

    def get_input_details(self):
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]

        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

    def get_output_details(self):
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]

    @staticmethod
    def rescale_boxes(boxes, input_shape, image_shape):
        # Rescale boxes to original image dimensions
        input_shape = np.array(
            [input_shape[1], input_shape[0], input_shape[1], input_shape[0]]
        )
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array(
            [image_shape[1], image_shape[0], image_shape[1], image_shape[0]]
        )

        return boxes
