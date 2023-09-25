
import cv2 as cv
import logging

from yolorun.lib.grabber import BBoxes

log = logging.getLogger(__name__)

class Model:
    segmentation = False

    def __init__(self, config):
        self.config = config
        #log.info("initialized model %s", self.__class__)
        _sizes = config.size.split("x")
        size = int(_sizes[0])
        if len(_sizes) > 1:
            self.size = [size, int(_sizes[1])]
        else:
            self.size = [size, size]
        self.bboxes = BBoxes(truth=False, segmentation=self.segmentation)
        self.w = 0
        self.h = 0
        self.gflops = 0

    def predict(self, frame):
        self.bboxes.reset()
        self.frame = frame
        self.frame_dirty = None
        self.h, self.w = frame.shape[:2]

    # def prepare_show(self):
    #     if self.frame_dirty is None:
    #         self.frame_dirty = self.frame.copy()

    # def draw_bboxes(self, bboxes):
    #     #if bboxes:
    #     bboxes.show(self.frame_dirty)

    def show(self, frame=None, scale=1.0):
        # if frame is None:
        #     frame = self.frame if self.frame_dirty is None else self.frame_dirty
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
        if "-seg" in config.model:
            print("TODO")
        else:
            from .ultralytics import ModelYolo
            return ModelYolo(config)
    elif ".cfg" in config.model:
        from .darknet import ModelDarknet
        return ModelDarknet(config) 
    elif ".onnx" in config.model:    
        if "-seg" in config.model:
            from .yoloseg_gorordo import ModelOnnxSeg
            return ModelOnnxSeg(config)
        else:
            if config.dnn:
                from .dnn import ModelDnn
                return ModelDnn(config)
            else:
                from .onnx import ModelOnnxRuntime
                return ModelOnnxRuntime(config)
                # from .onnx_dnn import ModelOnnxDnn
                # return ModelOnnxDnn(config)
    elif ".engine" in config.model:
        from .trt.pycuda import ModelTrt
        return ModelTrt(config) 
    return ModelDummy(config)

    log.error("no model for %s", config.model)
    