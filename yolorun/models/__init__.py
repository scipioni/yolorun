
import cv2 as cv
import logging

#from yolorun.grabber import BBoxes

log = logging.getLogger(__name__)

class Model:
    def __init__(self, config):
        self.config = config
        self.segmentation = False
        log.info("initialized model %s", self.__class__)

    def predict(self, frame):
        self.frame = frame
        self.frame_dirty = None

    def prepare_show(self):
        if self.frame_dirty is None:
            self.frame_dirty = self.frame.copy()

    def draw_bboxes(self, bboxes):
        if bboxes:
            bboxes.show(self.frame_dirty)

    def show(self, scale=1.0):
        frame = self.frame if self.frame_dirty is None else self.frame_dirty
        if scale != 1.0:
            h, w = frame.shape[:2]
            dim = (int(w * scale), int(h * scale))
            frame = cv.resize(frame, dim, interpolation=cv.INTER_AREA)
        cv.imshow(self.config.model, frame)

    def getBBoxes(self):
        return None
    #     return BBoxes(truth=False)


class ModelDummy(Model):
    pass

def getModel(config):
    if ".pt" in config.model:
        from .ultralytics import ModelYolo
        return ModelYolo(config)

    return ModelDummy(config)

    log.error("no model for %s", config.model)
    
