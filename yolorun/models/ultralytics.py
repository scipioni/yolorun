import cv2 as cv
from ultralytics import YOLO

from .__init__ import Model
from yolorun.grabber import BBoxes, BBox


class ModelYolo(Model):
    def __init__(self, config):
        super().__init__(config)

        self.net = YOLO(config.model)
        self.w = 0
        self.h = 0

    def predict(self, frame):
        super().predict(frame)
        self.h, self.w = frame.shape[:2]
        results = self.net(self.frame, verbose=False, stream=True)
        # self.boxes = results[0].boxes
        self.boxes = []
        for result in results:
            for box in result.boxes.cpu().numpy():
                if box.conf[0] > self.config.confidence_min:
                    self.boxes.append(box)

    def getBBoxes(self):
        bboxes = BBoxes(truth=False)
        for box in self.boxes:
            x1, y1, x2, y2 = box.xyxy[0].astype(int)[:4]
            bboxes.add(
                BBox(
                    box.cls[0],
                    x1,
                    y1,
                    x2,
                    y2,
                    self.w,
                    self.h,
                    box.conf[0]
                )
            )
        return bboxes

