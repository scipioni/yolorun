import cv2 as cv
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
import time
from numpy import ndarray
from typing import List, Tuple, Union

from yolorun.models import Model
from yolorun.lib.grabber import BBoxes, BBox
from yolorun.lib.timing import timing

def preproc(image, input_size, mean, std, swap=(2, 0, 1)):
    if len(image.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3)) * 114.0
    else:
        padded_img = np.ones(input_size) * 114.0
    img = np.array(image)
    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv.INTER_LINEAR,
    ).astype(np.float32)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img
    padded_img = padded_img[:, :, ::-1]
    padded_img /= 255.0
    if mean is not None:
        padded_img -= mean
    if std is not None:
        padded_img /= std
    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r

def preproc_fast(image, input_size, swap=(2, 0, 1)):
    padded_img = cv.resize(
        image,
        input_size,
        interpolation=cv.INTER_LINEAR,
    ).astype(np.float32)
    #padded_img = padded_img[:, :, ::-1]
    padded_img /= 255.0
    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    r = (input_size[0] / image.shape[0], input_size[1] / image.shape[1])
    return padded_img, r

def letterbox(im: ndarray,
              new_shape: Union[Tuple, List] = (640, 640),
              color: Union[Tuple, List] = (114, 114, 114)) \
        -> Tuple[ndarray, float, Tuple[float, float]]:
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    # new_shape: [width, height]

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[1], new_shape[1] / shape[0])
    # Compute padding [width, height]
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[0] - new_unpad[0], new_shape[1] - new_unpad[
        1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv.resize(im, new_unpad, interpolation=cv.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv.copyMakeBorder(im,
                            top,
                            bottom,
                            left,
                            right,
                            cv.BORDER_CONSTANT,
                            value=color)  # add border
    return np.ascontiguousarray(im, dtype=np.float32), r
    #return im, r, (dw, dh)

class ModelTrt(Model):
    def __init__(self, config):
        super().__init__(config)

        self.w = 0
        self.h = 0

        self.mean = None
        self.std = None
        
        logger = trt.Logger(trt.Logger.WARNING)
        logger.min_severity = trt.Logger.Severity.ERROR
        runtime = trt.Runtime(logger)
        trt.init_libnvinfer_plugins(logger,'') # initialize TensorRT plugins
        with open(config.model, "rb") as f:
            serialized_engine = f.read()
        engine = runtime.deserialize_cuda_engine(serialized_engine)
        self.imgsz = engine.get_binding_shape(0)[2:]  # get the read shape of model, in case user input it wrong
        self.context = engine.create_execution_context()
        self.inputs, self.outputs, self.bindings = [], [], []
        self.stream = cuda.Stream()
        for binding in engine:
            size = trt.volume(engine.get_binding_shape(binding))
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))
            if engine.binding_is_input(binding):
                self.inputs.append({'host': host_mem, 'device': device_mem})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem})
    
    def get_fps(self):
        img = np.ones((1,3,self.imgsz[0], self.imgsz[1]))
        img = np.ascontiguousarray(img, dtype=np.float32)
        for _ in range(5):  # warmup
            _ = self._infer(img)

        t0 = time.perf_counter()
        for _ in range(100):  # calculate average time
            _ = self._infer(img)
        return 100/(time.perf_counter() - t0)

    def __str__(self):
        return "  %s %.0ffps size=%sx%sx%d CUDA=%s" % (
            self.config.model,
            self.get_fps(),
            self.imgsz[0],
            self.imgsz[1],
            99,
            not self.config.cpu,
        )

    def predict(self, frame):
        super().predict(frame)

        with timing("preprocess"):
            #img, ratio = preproc(frame, self.imgsz, self.mean, self.std)
            img, ratio = preproc_fast(frame, self.imgsz)
            #img, ratio = letterbox(frame, self.imgsz)

        #img = np.ascontiguousarray(frame, dtype=np.float32)
        # resized = cv.resize(
        #     frame,
        #     (self.imgsz[1], self.imgsz[0]),
        #     interpolation=cv.INTER_LINEAR,
        #     ).astype(np.float32)
        # img = np.ascontiguousarray(resized, dtype=np.float32)
        with timing("inference"):
            data = self._infer(img)

        #ratio = min(self.imgsz[0] / frame.shape[0], self.imgsz[1] / frame.shape[1])

        with timing("postprocess"):
            num, final_boxes, final_scores, final_cls_inds = data
            #final_boxes = np.reshape(final_boxes/ratio, (-1, 4))
            final_boxes = np.reshape(final_boxes, (-1, 4))
            dets = np.concatenate([final_boxes[:num[0]], np.array(final_scores)[:num[0]].reshape(-1, 1), np.array(final_cls_inds)[:num[0]].reshape(-1, 1)], axis=-1)
            # self.h, self.w = frame.shape[:2]
            # results = self.net.predict(self.frame, imgsz=self.size, conf=self.config.confidence_min, verbose=False, stream=True)
            # for result in results:
            #     for box in result.boxes.cpu().numpy():
            #         if box.conf[0] > self.config.confidence_min:
            #             x1, y1, x2, y2 = box.xyxy[0].astype(int)[:4]
            #             self.bboxes.add(
            #                 BBox(box.cls[0], x1, y1, x2, y2, self.w, self.h, box.conf[0])
            #             )
            if dets is not None:
                boxes, confidences, classes = dets[:,:4], dets[:, 4], dets[:, 5]
                for i, box in enumerate(boxes):
                    if confidences[i] < self.config.confidence_min:
                        continue
                    left, top, right, bottom = box[:4]
                    self.bboxes.add(
                        BBox(
                            classes[i],
                            left / ratio[1],
                            top / ratio[0],
                            right / ratio[1],
                            bottom / ratio[0],
                            self.w,
                            self.h,
                            confidences[i],
                        )
                    )
            #print(final_boxes)

    def _infer(self, img):
            self.inputs[0]['host'] = np.ravel(img)
            # transfer data to the gpu
            for inp in self.inputs:
                cuda.memcpy_htod_async(inp['device'], inp['host'], self.stream)
            # run inference
            self.context.execute_async_v2(
                bindings=self.bindings,
                stream_handle=self.stream.handle)
            # fetch outputs from gpu
            for out in self.outputs:
                cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)
            # synchronize stream
            self.stream.synchronize()

            data = [out['host'] for out in self.outputs]
            return data