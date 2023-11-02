import math
import time
import cv2
import numpy as np
import onnxruntime
import logging

from .__init__ import Model
from yolorun.lib.grabber import BBoxes, BBox
from yolorun.lib.timing import timing

log = logging.getLogger(__name__)

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

# Create a list of colors for each class where each color is a tuple of 3 integer values
rng = np.random.default_rng(3)
colors = rng.uniform(0, 255, size=(len(class_names), 3))


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

        # print(keep_indices.shape, sorted_indices.shape)
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


def draw_detections(image, boxes, scores, class_ids, mask_alpha=0.3, mask_maps=None):
    img_height, img_width = image.shape[:2]
    size = min([img_height, img_width]) * 0.0006
    text_thickness = int(min([img_height, img_width]) * 0.001)

    mask_img = draw_masks(image, boxes, class_ids, mask_alpha, mask_maps)

    # Draw bounding boxes and labels of detections
    for box, score, class_id in zip(boxes, scores, class_ids):
        color = colors[class_id]

        x1, y1, x2, y2 = box.astype(int)

        # Draw rectangle
        cv2.rectangle(mask_img, (x1, y1), (x2, y2), color, 2)

        label = class_names[class_id]
        caption = f"{label} {int(score * 100)}%"
        (tw, th), _ = cv2.getTextSize(
            text=caption,
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=size,
            thickness=text_thickness,
        )
        th = int(th * 1.2)

        cv2.rectangle(mask_img, (x1, y1), (x1 + tw, y1 - th), color, -1)

        cv2.putText(
            mask_img,
            caption,
            (x1, y1),
            cv2.FONT_HERSHEY_SIMPLEX,
            size,
            (255, 255, 255),
            text_thickness,
            cv2.LINE_AA,
        )

    return mask_img


def draw_masks(image, boxes, class_ids, mask_alpha=0.3, mask_maps=None):
    mask_img = image.copy()

    # Draw bounding boxes and labels of detections
    for i, (box, class_id) in enumerate(zip(boxes, class_ids)):
        color = colors[class_id]

        x1, y1, x2, y2 = box.astype(int)

        # Draw fill mask image
        if mask_maps is None:
            cv2.rectangle(mask_img, (x1, y1), (x2, y2), color, -1)
        else:
            crop_mask = mask_maps[i][y1:y2, x1:x2, np.newaxis]
            crop_mask_img = mask_img[y1:y2, x1:x2]
            crop_mask_img = crop_mask_img * (1 - crop_mask) + crop_mask * color
            mask_img[y1:y2, x1:x2] = crop_mask_img

    return cv2.addWeighted(mask_img, mask_alpha, image, 1 - mask_alpha, 0)


def draw_comparison(img1, img2, name1, name2, fontsize=2.6, text_thickness=3):
    (tw, th), _ = cv2.getTextSize(
        text=name1,
        fontFace=cv2.FONT_HERSHEY_DUPLEX,
        fontScale=fontsize,
        thickness=text_thickness,
    )
    x1 = img1.shape[1] // 3
    y1 = th
    offset = th // 5
    cv2.rectangle(
        img1,
        (x1 - offset * 2, y1 + offset),
        (x1 + tw + offset * 2, y1 - th - offset),
        (0, 115, 255),
        -1,
    )
    cv2.putText(
        img1,
        name1,
        (x1, y1),
        cv2.FONT_HERSHEY_DUPLEX,
        fontsize,
        (255, 255, 255),
        text_thickness,
    )

    (tw, th), _ = cv2.getTextSize(
        text=name2,
        fontFace=cv2.FONT_HERSHEY_DUPLEX,
        fontScale=fontsize,
        thickness=text_thickness,
    )
    x1 = img2.shape[1] // 3
    y1 = th
    offset = th // 5
    cv2.rectangle(
        img2,
        (x1 - offset * 2, y1 + offset),
        (x1 + tw + offset * 2, y1 - th - offset),
        (94, 23, 235),
        -1,
    )

    cv2.putText(
        img2,
        name2,
        (x1, y1),
        cv2.FONT_HERSHEY_DUPLEX,
        fontsize,
        (255, 255, 255),
        text_thickness,
    )

    combined_img = cv2.hconcat([img1, img2])
    if combined_img.shape[1] > 3840:
        combined_img = cv2.resize(combined_img, (3840, 2160))

    return combined_img


class YOLOSeg:
    def __init__(self, path, conf_thres=0.7, iou_thres=0.5, num_masks=32):
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres
        self.num_masks = num_masks

        # Initialize model
        self.initialize_model(path)

    def __call__(self, image):
        return self.segment_objects(image)

    def initialize_model(self, path):
        self.session = onnxruntime.InferenceSession(
            path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        # Get model info
        self.get_input_details()
        self.get_output_details()
        print("onnx device:", onnxruntime.get_device())

    def segment_objects(self, image):
        with timing("preprocess"):
            input_tensor = self.prepare_input(image)

        # Perform inference on the image
        with timing("inference"):
            outputs = self.inference(input_tensor)

        with timing("postprocess"):
            self.boxes, self.scores, self.class_ids, mask_pred = self.process_box_output(
                outputs[0]
            )
            self.mask_maps = self.process_mask_output(mask_pred, outputs[1])

        return self.boxes, self.scores, self.class_ids, self.mask_maps

    def prepare_input(self, image):
        self.img_height, self.img_width = image.shape[:2]

        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize input image
        input_img = cv2.resize(input_img, (self.input_width, self.input_height))

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
            return []

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
            int(self.img_width / mask_width),
            int(self.img_height / mask_height),
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
            crop_mask = cv2.resize(
                scale_crop_mask, (x2 - x1, y2 - y1), interpolation=cv2.INTER_CUBIC
            )

            crop_mask = cv2.blur(crop_mask, blur_size)

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

    def draw_detections(self, image, draw_scores=True, mask_alpha=0.4):
        return draw_detections(
            image, self.boxes, self.scores, self.class_ids, mask_alpha
        )

    def draw_masks(self, image, draw_scores=True, mask_alpha=0.5):
        return draw_detections(
            image,
            self.boxes,
            self.scores,
            self.class_ids,
            mask_alpha,
            mask_maps=self.mask_maps,
        )

    def get_input_details(self):
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]

        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]
        print(f"model input: {self.input_width}x{self.input_height}")

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


class ModelOnnxSeg(Model):
    """
     input_yolov8-seg:
         NodeArg(name='images', type='tensor(float)', shape=[1, 3, 640, 640])
     output_yolov8-seg:
         NodeArg(name='output0', type='tensor(float)', shape=[1, 116, 8400])
         NodeArg(name='output1', type='tensor(float)', shape=[1, 32, 160, 160])


     input_nms:
         NodeArg(name='detection', type='tensor(float)', shape=[1, None, None])
         NodeArg(name='config', type='tensor(float)', shape=[4])
     output_nms:
         NodeArg(name='selected', type='tensor(float)', shape=[1, 'unk__4', 'unk__1'])

    mask-yolov8:
     - input
         - NodeArg(name='detection', type='tensor(float)', shape=[None])
         - NodeArg(name='mask', type='tensor(float)', shape=[1, None, None, None])
         - NodeArg(name='config', type='tensor(float)', shape=[9])
         - NodeArg(name='overlay', type='tensor(uint8)', shape=[None, None, 4])
     - output
         - NodeArg(name='mask_filter', type='tensor(uint8)', shape=[None, None, 4])
    """

    segmentation = True
    num_masks = 32

    def __init__(self, config):
        super().__init__(config)

        self.yoloseg = YOLOSeg(config.model, conf_thres=0.5, iou_thres=0.7)
        if self.config.model_nms:
            log.info("loading nms model %s", self.config.model_nms)

            self.session_nms = onnxruntime.InferenceSession(
                self.config.model_nms,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            self.config_nms = np.array(
                [
                    80,  # numclasses
                    100,  # topk
                    0.7,  # iou_thresh
                    self.config.confidence_min,  # score_thresh
                ]
            ).astype(np.float32)

            self.session_mask = onnxruntime.InferenceSession(
                self.config.model_mask,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )

    def predict(self, frame):
        super().predict(frame)

        if self.config.model_nms:
            self._predict_with_nms(frame)
        else:
            boxes, scores, class_ids, masks = self.yoloseg(frame)

            for i, box in enumerate(boxes):
                left, top, right, bottom = box
                if scores[i] < self.config.confidence_min:
                    continue

                self.bboxes.add(
                    BBox(
                        class_ids[i],
                        left,
                        top,
                        right,
                        bottom,
                        self.w,
                        self.h,
                        scores[i],
                        mask=masks[i],
                    )
                )

        # combined_img = yoloseg.draw_masks(img)

    def _predict_with_nms(self, frame):
        with timing("preprocess"):
            input_tensor = self.yoloseg.prepare_input(frame)

        with timing("inference"):
            output0, output1 = self.yoloseg.session.run(
                self.yoloseg.output_names, {self.yoloseg.input_names[0]: input_tensor}
            )

        with timing("postprocess"):
            selected = self.session_nms.run(
                ["selected"], {"detection": output0, "config": self.config_nms}
            )

            selected = np.array(selected)
            predictions = np.squeeze(selected, axis=(0,1))
            num_classes = predictions.shape[1] - self.num_masks - 4
            scores = np.max(predictions[:, 4 : 4 + num_classes], axis=1)
            #predictions = predictions[scores > self.config.confidence_min, :]
            # scores = scores[scores > self.config.confidence_min]
            box_predictions = predictions[..., : num_classes + 4]
            class_ids = np.argmax(box_predictions[:, 4:], axis=1)
            boxes = self.yoloseg.extract_boxes(box_predictions)
            for i, box in enumerate(boxes):
                box_on_model = predictions[i, :4]
                box_on_model[0] = box_on_model[0] - 0.5*box_on_model[2]
                box_on_model[1] = box_on_model[1] - 0.5*box_on_model[3]

                mask_filter = self._get_mask(
                    output1=output1,
                    box_on_model=box_on_model,
                    mask=predictions[i, 4 + num_classes :],
                )
                mask2 = cv2.resize(cv2.cvtColor(mask_filter, cv2.COLOR_BGRA2BGR), (self.w, self.h))
                left, top, right, bottom = box
                self.bboxes.add(
                    BBox(
                        class_ids[i],
                        left,
                        top,
                        right,
                        bottom,
                        self.w,
                        self.h,
                        scores[i],
                        mask2=mask2,
                    )
                )

    def _get_mask(self, output1, box_on_model, mask):
        mask = np.concatenate((box_on_model, mask)).astype(np.float32)
        max_size = max(self.yoloseg.input_height, self.yoloseg.input_width)
        left, top, w, h = box_on_model
        mask_config = np.array(
            [
                max_size,
                left,
                top,
                w,
                h,
                255, 
                120,
                120,
                120,  # ...Colors.hexToRgba(color, 120), // color in RGBA
            ]
        ).astype(np.float32)
        overlay = np.zeros((self.yoloseg.input_width, self.yoloseg.input_height, 4), dtype=np.uint8)

        mask_filter = self.session_mask.run(
            ["mask_filter"],
            {
                "detection": mask,
                "mask": output1,
                "config": mask_config,
                "overlay": overlay,
            },
        )
        return np.squeeze(mask_filter)

    def _draw_masks(self, frame, masks):
        mask_img = frame.copy()
        for i, box in enumerate(self.bboxes.get()):
            color = (0, 255, 0)
            x1, y1, x2, y2 = map(int, box.box)
            crop_mask = masks[i][y1:y2, x1:x2, np.newaxis]
            crop_mask_img = mask_img[y1:y2, x1:x2]
            crop_mask_img = crop_mask_img * (1 - crop_mask) + crop_mask * color
            mask_img[y1:y2, x1:x2] = crop_mask_img
        self.frame_dirty = cv2.addWeighted(
            mask_img, mask_alpha, self.frame_dirty, 1 - mask_alpha, 0
        )

    # def show(self, scale=1.0, mask_alpha=0.3):
    #     self.frame_dirty = self._draw_masks(self.frame)
    #     super().show(scale)
