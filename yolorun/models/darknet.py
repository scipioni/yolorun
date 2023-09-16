import cv2 as cv
from .__init__ import Model
from yolorun.lib.grabber import BBoxes, BBox


class ModelDarknet(Model):
    def __init__(self, config):
        super().__init__(config)

        if not config.weights:
            config.weights = config.model.replace(".cfg", ".weights")

        _net = cv.dnn.readNet(config.model, config.weights)

        self.net = cv.dnn_DetectionModel(_net)
        self.net.setInputParams(
            size=self.size,
            mean=(0, 0, 0),
            scale=1.0 / 255.0,
            swapRB=True,
            crop=config.crop,
        )
        self.channels = _net.getParam(_net.getLayerNames()[0]).shape[1]
        self.gflops = (
            _net.getFLOPS((1, self.channels, self.size[0], self.size[1])) * 1e-9
        )
        if not config.cpu:
            self.net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)

        self.w = 0
        self.h = 0

    def __str__(self):
        return "  %s weigths=%s %.2fGFLOP size=%sx%sx%d CUDA=%s" % (
            self.config.model,
            self.config.weights,
            self.gflops,
            self.size[0],
            self.size[1],
            self.channels,
            not self.config.cpu,
        )

    def predict(self, frame):
        super().predict(frame)
        self.h, self.w = frame.shape[:2]

        if len(frame.shape) > 2 and self.channels == 1:  # rete con un canale solo
            frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        confThreshold = 0.1
        nmsThreshold = 0.4

        classIds, confidences, boxes = self.net.detect(
            frame, confThreshold, nmsThreshold
        )

       

        for i, box in enumerate(boxes):
            left, top, width, height = box
            if confidences[i] < self.config.confidence_min:
                continue

            self.bboxes.add(
                BBox(
                    classIds[i],
                    left,
                    top,
                    left + width,
                    top + height,
                    self.w,
                    self.h,
                    confidences[i],
                )
                # {
                #     "classId": classIds[i],
                #     "confidence": confidences[i],
                #     "box": (left, top, width, height),
                #     "name": classIds[i], # self.classes[int(classIds[i])],
                #     "frame_shape": frame.shape,
                # }
            )

    def getBBoxes(self):
        return self.bboxes
        # bboxes = BBoxes(truth=False)
        # for box in self.boxes:
        #     x1, y1, x2, y2 = box.xyxy[0].astype(int)[:4]
        #     bboxes.add(BBox(box.cls[0], x1, y1, x2, y2, self.w, self.h, box.conf[0]))
        # return bboxes
