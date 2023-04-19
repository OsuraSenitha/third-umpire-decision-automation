from typing import Tuple
import onnxruntime
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def xywh2xyxy(x):
    # Convert bounding box (x, y, w, h) to bounding box (x1, y1, x2, y2)
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y

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

def annotate_image(image, boxes, scores, labels):
    output_image = image.copy()
    for (bbox, score, cls) in zip(boxes, scores, labels):
        bbox = bbox.round().astype(np.int32).tolist()
        color = (0,255,0)
        cv.rectangle(output_image, tuple(bbox[:2]), tuple(bbox[2:]), color, 2)
        cv.putText(output_image,
                    f'{cls}:{int(score*100)}', (bbox[0], bbox[1] - 2),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.60, [225, 255, 255],
                    thickness=1)
        
    return output_image

class ObjectDetectOutput:
    def __init__(self, boxes, scores, labels, image) -> None:
        self.boxes = boxes
        self.scores = scores
        self.labels = labels
        self.image = image

    def get_image(self, annotate: bool = True) -> np.ndarray:
        image = self.image
        if annotate:
            image = annotate_image(image, self.boxes, self.scores, self.labels)
        return image
    
    def show_image(self, annotate: bool = True) -> None:
        image = self.get_image(annotate)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        plt.imshow(image)
        plt.show()

    def tolist(self) -> Tuple[Tuple]:
        boxes = self.boxes
        labels = self.labels
        scores = self.scores

        labels = np.array(labels)
        labels_c = labels.reshape((-1, 1))
        scores_c = scores.reshape((-1, 1))
        output = np.hstack((labels_c, boxes, scores_c))
        output = output.tolist()

        return output

class ObjectDetectModel:
    '''
    Detects the objects in a given image
    Source: https://alimustoofaa.medium.com/how-to-load-model-yolov8-onnx-runtime-b632fad33cec
    '''
    def __init__(self, onnx_path: str, classes: Tuple[str]=["Batsmen", "Ball", "Wicket"]) -> None:
        EP_list = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        ort_session = onnxruntime.InferenceSession(onnx_path, providers=EP_list)
        
        model_inputs = ort_session.get_inputs()
        input_names = [model_inputs[i].name for i in range(len(model_inputs))]
        input_shapes = [model_inputs[i].shape for i in range(len(model_inputs))]
        input_types = [model_inputs[i].type for i in range(len(model_inputs))]
        input_format = {k:{"shape": s, "type": t} for k, (s, t) in zip(input_names, zip(input_shapes, input_types))}

        model_output = ort_session.get_outputs()
        output_names = [model_output[i].name for i in range(len(model_output))]
        output_shapes = [model_output[i].shape for i in range(len(model_output))]
        output_types = [model_output[i].type for i in range(len(model_output))]
        output_format = {k:{"shape": s, "type": t} for k, (s, t) in zip(output_names, zip(output_shapes, output_types))}

        self.ort_session = ort_session
        self.classes = classes
        self.input_shape = input_shapes[0]
        self.output_names = output_names
        self.input_name = input_names[0]

        self.input_format = input_format
        self.output_format = output_format

    def __call__(self, img_path: str, conf_threshold: int = 0.8) -> ObjectDetectOutput:
        image = cv.imread(img_path)
        image_height, image_width = image.shape[:2]
        
        input_height, input_width = self.input_shape[2:]
        image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        resized = cv.resize(image_rgb, (input_width, input_height))

        # Scale input pixel value to 0 to 1
        input_image = resized / 255.0
        input_image = input_image.transpose(2,0,1)
        input_tensor = input_image[np.newaxis, :, :, :].astype(np.float32)

        outputs = self.ort_session.run(self.output_names, {self.input_name: input_tensor})[0]

        predictions = np.squeeze(outputs).T
        # Filter out object confidence scores below threshold
        scores = np.max(predictions[:, 4:], axis=1)
        predictions = predictions[scores > conf_threshold, :]
        scores = scores[scores > conf_threshold]

        # Get the class with the highest confidence
        class_ids = np.argmax(predictions[:, 4:], axis=1).astype(int)

        # Get bounding boxes for each object
        boxes = predictions[:, :4]

        #rescale box
        input_shape = np.array([input_width, input_height, input_width, input_height])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([image_width, image_height, image_width, image_height])
        boxes = boxes.astype(np.int32)

        # convert box format
        boxes = xywh2xyxy(boxes)

        # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
        indices = nms(boxes, scores, 0.3)

        boxes, scores, class_ids = boxes[indices], scores[indices], class_ids[indices]
        labels = list(map(lambda id: self.classes[id], class_ids))

        output = ObjectDetectOutput(boxes, scores, labels, image)

        return output
