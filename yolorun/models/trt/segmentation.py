import cv2 as cv
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
import time
from numpy import ndarray
from typing import List, Tuple, Union
import json
from collections import OrderedDict, namedtuple
#import torch
from yolorun.models import Model
from yolorun.lib.grabber import BBoxes, BBox
from yolorun.lib.timing import timing

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv.resize(im, new_unpad, interpolation=cv.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv.copyMakeBorder(im, top, bottom, left, right, cv.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)

def sigmoid(x): 
    return 1.0/(1+np.exp(-x))

def process_mask(protos, masks_in, bboxes, shape):

    c, mh, mw = protos.shape  # CHW
    ih, iw = shape
    masks = sigmoid(masks_in @ protos.reshape(c, -1)).reshape(-1, mh, mw)  # CHW 【lulu】

    downsampled_bboxes = bboxes.copy()
    downsampled_bboxes[:, 0] *= mw / iw
    downsampled_bboxes[:, 2] *= mw / iw
    downsampled_bboxes[:, 3] *= mh / ih
    downsampled_bboxes[:, 1] *= mh / ih

    masks = crop_mask(masks, downsampled_bboxes)  # CHW
    masks = np.transpose(masks, [1,2,0])
    # masks = cv.resize(masks, (shape[1], shape[0]), interpolation=cv.INTER_NEAREST)
    masks = cv.resize(masks, (shape[1], shape[0]), interpolation=cv.INTER_LINEAR)
    if masks.ndim == 3:
        masks = np.transpose(masks, [2,0,1])
    return np.where(masks>0.5,masks,0)

def preprocess(image, input_height, input_width):
    image_3c = image

    # Convert the image_3c color space from BGR to RGB
    image_3c = cv.cvtColor(image_3c, cv.COLOR_BGR2RGB)

    # Resize the image_3c to match the input shape
    image_3c, ratio, dwdh = letterbox(image_3c, new_shape=[input_height, input_width], auto=False)

    # Normalize the image_3c data by dividing it by 255.0
    image_4c = np.array(image_3c) / 255.0

    # Transpose the image_3c to have the channel dimension as the first dimension
    image_4c = np.transpose(image_4c, (2, 0, 1))  # Channel first

    # Expand the dimensions of the image_3c data to match the expected input shape
    image_4c = np.expand_dims(image_4c, axis=0).astype(np.float32)

    image_4c = np.ascontiguousarray(image_4c)  # contiguous
    
    # Return the preprocessed image_3c data
    return image_4c, image_3c

def non_max_suppression(
        prediction,
        conf_thres=0.25,
        iou_thres=0.45,
        classes=None,
        agnostic=False,
        multi_label=False,
        labels=(),
        max_det=300,
        nc=0,  # number of classes (optional)
):

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'
    
    #【lulu】prediction.shape[1]：box + cls + num_masks
    bs = prediction.shape[0]              # batch size
    nc = nc or (prediction.shape[1] - 4)  # number of classes
    nm = prediction.shape[1] - nc - 4     # num_masks
    mi = 4 + nc                           # mask start index
    xc = np.max(prediction[:, 4:mi], axis=1) > conf_thres ## 【lulu】

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    max_wh = 7680  # (pixels) maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 0.5 + 0.05 * bs  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [np.zeros((0,6 + nm))] * bs ## 【lulu】

    for xi, x in enumerate(prediction):  # image_3c index, image_3c inference
        # Apply constraints
        # x[((x[:, 2:4] < min_wh) | (x[:, 2:4] > max_wh)).any(1), 4] = 0  # width-height

        x = np.transpose(x,[1,0])[xc[xi]] ## 【lulu】
        # If none remain process next image_3c
        if not x.shape[0]: continue

        # Detections matrix nx6 (xyxy, conf, cls)
        box, cls, mask = np.split(x, [4, 4+nc], axis=1) ## 【lulu】
        box = xywh2xyxy(box)  # center_x, center_y, width, height) to (x1, y1, x2, y2)

        j = np.argmax(cls, axis=1)  ## 【lulu】
        conf = cls[np.array(range(j.shape[0])), j].reshape(-1,1)
        x = np.concatenate([box, conf, j.reshape(-1,1), mask], axis=1)[conf.reshape(-1,)>conf_thres]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n: continue
        x = x[np.argsort(x[:, 4])[::-1][:max_nms]]  # sort by confidence and remove excess boxes 【lulu】
        # Batched NMS
        c = x[:, 5:6] * max_wh  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = nms(boxes, scores, iou_thres) ## 【lulu】
        i = i[:max_det]  # limit detections

        output[xi] = x[i]

        if (time.time() - t) > time_limit:
            # LOGGER.warning(f'WARNING ⚠️ NMS time limit {time_limit:.3f}s exceeded')
            break  # time limit exceeded
    return output

def postprocess(preds, orig_img, OBJ_THRESH=0.25, NMS_THRESH=0.45, classes=None):
    p = non_max_suppression(preds[0],
                                OBJ_THRESH,
                                NMS_THRESH,
                                agnostic=False,
                                max_det=300,
                                nc=classes,
                                classes=None)        
    #print(p)                    
    results = []
    proto = preds[1]

    #hwc = img.shape[2:]
    hwc = orig_img.shape[:3]
    
    for i, pred in enumerate(p):
        shape = orig_img.shape
        if not len(pred):
            results.append([[], [], []])  # save empty boxes
            continue
        masks = process_mask(proto[i], pred[:, 6:], pred[:, :4], hwc)  # HWC
        pred[:, :4] = scale_boxes(hwc, pred[:, :4], shape).round()
        results.append([pred[:, :6], masks, shape[:2]])
    return results

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
class ModelTrtSegmentation(Model):
    def __init__(self, config):
        super().__init__(config)

        self.w = 0
        self.h = 0

        ##### with torch
        # Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
        # logger = trt.Logger(trt.Logger.INFO)
        # device = torch.device('cuda:0')
        # # Read file
        # with open(config.model, 'rb') as f, trt.Runtime(logger) as runtime:
        #     meta_len = int.from_bytes(f.read(4), byteorder='little')  # read metadata length
        #     metadata = json.loads(f.read(meta_len).decode('utf-8'))  # read metadata
        #     model = runtime.deserialize_cuda_engine(f.read())  # read engine
        # context = model.create_execution_context()
        # bindings = OrderedDict()
        # input_names = []
        # output_names = []
        # for i in range(model.num_bindings):
        #     name = model.get_binding_name(i) 
        #     dtype = trt.nptype(model.get_binding_dtype(i))
        #     if model.binding_is_input(i):
        #         input_names.append(name)
        #     else:  # output
        #         output_names.append(name)
        #     shape = tuple(context.get_binding_shape(i))
        #     im = torch.from_numpy(np.empty(shape, dtype=dtype)).to(device)
        #     bindings[name] = Binding(name, dtype, shape, im, int(im.data_ptr()))
        # binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())


        self.mean = None
        self.std = None
        
        logger = trt.Logger(trt.Logger.INFO)
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

        #image_4c, image_3c = preprocess(frame, self.size[1], self.size[0])
        #image_4c = image_4c.astype(np.float32)
        #image_4c = torch.from_numpy(image_4c).to(device)


        with timing("preprocess"):
            img, ratio = preproc_fast(frame, self.imgsz)
    #         #img, ratio = letterbox(frame, self.imgsz)

        with timing("inference"):
            data = self._infer(img)


        out1, out2 = data
        print(out1.shape, out2.shape)
        results = postprocess(data, img, classes=80) ##[box,mask,shape]

    #     with timing("postprocess"):
    #         num, final_boxes, final_scores, final_cls_inds = data
    #         #final_boxes = np.reshape(final_boxes/ratio, (-1, 4))
    #         final_boxes = np.reshape(final_boxes, (-1, 4))
    #         dets = np.concatenate([final_boxes[:num[0]], np.array(final_scores)[:num[0]].reshape(-1, 1), np.array(final_cls_inds)[:num[0]].reshape(-1, 1)], axis=-1)
    #         # self.h, self.w = frame.shape[:2]
    #         # results = self.net.predict(self.frame, imgsz=self.size, conf=self.config.confidence_min, verbose=False, stream=True)
    #         # for result in results:
    #         #     for box in result.boxes.cpu().numpy():
    #         #         if box.conf[0] > self.config.confidence_min:
    #         #             x1, y1, x2, y2 = box.xyxy[0].astype(int)[:4]
    #         #             self.bboxes.add(
    #         #                 BBox(box.cls[0], x1, y1, x2, y2, self.w, self.h, box.conf[0])
    #         #             )
    #         if dets is not None:
    #             boxes, confidences, classes = dets[:,:4], dets[:, 4], dets[:, 5]
    #             for i, box in enumerate(boxes):
    #                 if confidences[i] < self.config.confidence_min:
    #                     continue
    #                 left, top, right, bottom = box[:4]
    #                 self.bboxes.add(
    #                     BBox(
    #                         classes[i],
    #                         left / ratio[1],
    #                         top / ratio[0],
    #                         right / ratio[1],
    #                         bottom / ratio[0],
    #                         self.w,
    #                         self.h,
    #                         confidences[i],
    #                     )
    #                 )
    #         #print(final_boxes)

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