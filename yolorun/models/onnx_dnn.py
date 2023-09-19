# https://github.com/ultralytics/ultralytics/blob/main/examples/YOLOv8-OpenCV-ONNX-Python/main.py

import cv2 as cv
import numpy as np

from .__init__ import Model
from yolorun.lib.grabber import BBoxes, BBox
from yolorun.lib.timing import timing

class ModelOnnxDnn(Model):
    def __init__(self, config):
        super().__init__(config)

        self.net: cv.dnn.Net = cv.dnn.readNetFromONNX(config.model)
        
        if not config.cpu:
            self.net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)

        self.channels = self.net.getParam(self.net.getLayerNames()[0]).shape[1]
        self.gflops = (
            self.net.getFLOPS((1, self.channels, self.size[0], self.size[1])) * 1e-9
        )


    def __str__(self):
        return "  %s %.2fGFLOP size=%sx%sx%d CUDA=%s" % (
            self.config.model,
            self.gflops,
            self.size[0],
            self.size[1],
            self.channels,
            not self.config.cpu,
        )

    def predict(self, frame):
        confThreshold = 0.25
        nmsThreshold = 0.45
        eta = 0.5

        super().predict(frame)
        if len(frame.shape) > 2 and self.channels == 1:  # rete con un canale solo
            frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        
        with timing("cv.dnn.forward"):
            blob = cv.dnn.blobFromImage(frame, scalefactor=1 / 255, size=self.size, swapRB=True)
            self.net.setInput(blob)
            outputs = self.net.forward()

        with timing("filter confidence"):
            scale_w = frame.shape[1] / self.size[0]
            scale_h = frame.shape[0] / self.size[1]
            outputs = np.array([cv.transpose(outputs[0])])
            rows = outputs.shape[1]

            boxes = []
            scores = []
            class_ids = []

            for i in range(rows):
                classes_scores = outputs[0][i][4:]
                (minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv.minMaxLoc(classes_scores)
                if maxScore >= self.config.confidence_min:
                    box = [
                        outputs[0][i][0] - (0.5 * outputs[0][i][2]), outputs[0][i][1] - (0.5 * outputs[0][i][3]),
                        outputs[0][i][2], outputs[0][i][3]]
                    boxes.append(box)
                    scores.append(maxScore)
                    class_ids.append(maxClassIndex)

        with timing("nms"):
            result_boxes = cv.dnn.NMSBoxes(boxes, scores, confThreshold, nmsThreshold, eta)

        detections = []
        for i in range(len(result_boxes)):
            index = result_boxes[i]
            box = boxes[index]
            left, top, width, height = box
            
            self.bboxes.add(
                BBox(
                    class_ids[index],
                    left * scale_w,
                    top * scale_h,
                    (left + width)*scale_w,
                    (top + height)*scale_h,
                    self.w,
                    self.h,
                    scores[index],
                )
            )



# TODO

#   def _nms(self, detect_results):
#     """Apply non-maximum suppression and filter detection results
#     :param detect_results: List[DetectResult]
#     :returns List[DetectResult]
#     """
#     confidences = [float(d.conf) for d in detect_results]
#     boxes = [d.bbox.xywh_int for d in detect_results]
#     idxs = cv.dnn.NMSBoxes(boxes, confidences, self.dnn_cfg.threshold, self.dnn_cfg.nms_threshold)
#     detect_results_nms = [detect_results[i] for i in idxs]
#     return detect_results_nms