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
            )

