import cv2 as cv
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
import time

from .__init__ import Model
from yolorun.lib.grabber import BBoxes, BBox


class ModelTrt(Model):
    def __init__(self, config):
        super().__init__(config)

        self.w = 0
        self.h = 0

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
        #img = np.ascontiguousarray(frame, dtype=np.float32)
        resized = cv.resize(
            frame,
            (self.imgsz[1], self.imgsz[0]),
            interpolation=cv.INTER_LINEAR,
            ).astype(np.float32)
        img = np.ascontiguousarray(resized, dtype=np.float32)
        data = self._infer(img)

        ratio = min(self.imgsz[0] / frame.shape[0], self.imgsz[1] / frame.shape[1])

        num, final_boxes, final_scores, final_cls_inds = data
        final_boxes = np.reshape(final_boxes/ratio, (-1, 4))
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
            final_boxes, final_scores, final_cls_inds = dets[:,:4], dets[:, 4], dets[:, 5]
            print(final_boxes)

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