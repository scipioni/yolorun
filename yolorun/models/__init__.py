
import cv2 as cv
import logging

from yolorun.lib.grabber import BBoxes

log = logging.getLogger(__name__)

class Model:
    def __init__(self, config):
        self.config = config
        self.segmentation = False
        log.info("initialized model %s", self.__class__)
        _sizes = config.size.split("x")
        size = int(_sizes[0])
        if len(_sizes) > 1:
            self.size = [size, int(_sizes[1])]
        else:
            self.size = [size, size]
        self.bboxes = BBoxes(truth=False)

    def predict(self, frame):
        self.bboxes.reset()
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
        return self.bboxes
class ModelDummy(Model):
    pass

def getModel(config):
    if ".pt" in config.model:
        from .ultralytics import ModelYolo
        return ModelYolo(config)

    elif ".cfg" in config.model:
        from .darknet import ModelDarknet
        return ModelDarknet(config) 
    
    return ModelDummy(config)

    log.error("no model for %s", config.model)
    
