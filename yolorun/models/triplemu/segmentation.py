import cv2 as cv
import numpy as np
from numpy import ndarray
import torch
from typing import List, Tuple, Union
from torch import Tensor
import torch.nn.functional as F
from torchvision.ops import batched_nms
from yolorun.models import Model
from yolorun.lib.grabber import BBoxes, BBox
from yolorun.lib.timing import timing

from .engine import TRTModule

class TripleMuSegmentation(Model):
    segmentation = True

    def __init__(self, config):
        super().__init__(config)
        self.device = torch.device(config.device)

        self.engine = TRTModule(config.model, self.device)
        self.H, self.W = self.engine.inp_info[0].shape[-2:]

        # set desired output names order
        self.engine.set_desired(['outputs', 'proto'])



    def predict(self, frame):
        super().predict(frame)

        with timing("preprocess"):
            bgr, ratio, dwdh = letterbox(frame, (self.W, self.H))
            dw, dh = int(dwdh[0]), int(dwdh[1])
            rgb = cv.cvtColor(bgr, cv.COLOR_BGR2RGB)

            tensor, seg_img = blob(rgb, return_seg=True)
            dwdh = torch.asarray(dwdh * 2, dtype=torch.float32, device=self.device)
            tensor = torch.asarray(tensor, device=self.device)

        with timing("inference"):
            data = self.engine(tensor)

        with timing("postprocess"):
            seg_img = torch.asarray(seg_img[dh:self.H - dh, dw:self.W - dw, [2, 1, 0]],
                                    device=self.device)
            bboxes, scores, labels, masks = seg_postprocess(
                data, bgr.shape[:2], self.config.confidence_min, self.config.iou_thres)

            if bboxes.numel() == 0:
                # if no bounding box
                return
                
            masks = masks[:, dh:self.H - dh, dw:self.W - dw, :]
            indices = (labels % len(MASK_COLORS)).long()
            bboxes -= dwdh
            bboxes /= ratio
        #mask_colors = torch.asarray(MASK_COLORS, device=self.device)[indices]
        #mask_colors = mask_colors.view(-1, 1, 1, 3) * ALPHA
        #mask_colors = masks @ mask_colors
        #inv_alph_masks = (1 - masks * 0.5).cumprod(0)
        #mcs = (mask_colors * inv_alph_masks).sum(0) * 2
        #seg_img = (seg_img * inv_alph_masks[-1] + mcs) * 255

        #draw = cv.resize(seg_img.cpu().numpy().astype(np.uint8), frame.shape[:2][::-1])
        #cv.imshow("mask", draw)
        for (bbox, score, label, mask) in zip(bboxes, scores, labels, masks):
            if score < self.config.confidence_min:
                continue
            left,top,right,bottom = bbox.round().int().tolist()
            cls_id = int(label)
            mask = cv.resize(mask.cpu().numpy().astype(np.uint8), frame.shape[:2][::-1])
            self.bboxes.add(
                    BBox(
                        cls_id,
                        left,
                        top,
                        right,
                        bottom,
                        self.w,
                        self.h,
                        score.item(),
                        mask=mask,
                    )
                )



    def __str__(self):
        return "  %s %.2fGFLOP size=%sx%sx%d CUDA=%s" % (
            self.config.model,
            0, #self.gflops,
            self.W,
            self.H,
            0, #self.channels,
            not self.config.cpu,
        )    


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
    return im, r, (dw, dh)

def blob(im: ndarray, return_seg: bool = False) -> Union[ndarray, Tuple]:
    seg = None
    if return_seg:
        seg = im.astype(np.float32) / 255
    im = im.transpose([2, 0, 1])
    im = im[np.newaxis, ...]
    im = np.ascontiguousarray(im).astype(np.float32) / 255
    if return_seg:
        return im, seg
    else:
        return im


def seg_postprocess(
        data: Tuple[Tensor],
        shape: Union[Tuple, List],
        conf_thres: float = 0.25,
        iou_thres: float = 0.65) \
        -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    assert len(data) == 2
    h, w = shape[0] // 4, shape[1] // 4  # 4x downsampling
    outputs, proto = data[0][0], data[1][0]
    bboxes, scores, labels, maskconf = outputs.split([4, 1, 1, 32], 1)
    scores, labels = scores.squeeze(), labels.squeeze()
    idx = scores > conf_thres
    if not idx.any():  # no bounding boxes or seg were created
        return bboxes.new_zeros((0, 4)), scores.new_zeros(
            (0, )), labels.new_zeros((0, )), bboxes.new_zeros((0, 0, 0, 0))
    bboxes, scores, labels, maskconf = \
        bboxes[idx], scores[idx], labels[idx], maskconf[idx]
    idx = batched_nms(bboxes, scores, labels, iou_thres)
    bboxes, scores, labels, maskconf = \
        bboxes[idx], scores[idx], labels[idx].int(), maskconf[idx]
    masks = (maskconf @ proto).sigmoid().view(-1, h, w)
    masks = crop_mask(masks, bboxes / 4.)
    masks = F.interpolate(masks[None],
                          shape,
                          mode='bilinear',
                          align_corners=False)[0]
    masks = masks.gt_(0.5)[..., None]
    return bboxes, scores, labels, masks


def crop_mask(masks: Tensor, bboxes: Tensor) -> Tensor:
    n, h, w = masks.shape
    x1, y1, x2, y2 = torch.chunk(bboxes[:, :, None], 4, 1)  # x1 shape(1,1,n)
    r = torch.arange(w, device=masks.device,
                     dtype=x1.dtype)[None, None, :]  # rows shape(1,w,1)
    c = torch.arange(h, device=masks.device,
                     dtype=x1.dtype)[None, :, None]  # cols shape(h,1,1)

    return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))


MASK_COLORS = np.array([(255, 56, 56), (255, 157, 151), (255, 112, 31),
                        (255, 178, 29), (207, 210, 49), (72, 249, 10),
                        (146, 204, 23), (61, 219, 134), (26, 147, 52),
                        (0, 212, 187), (44, 153, 168), (0, 194, 255),
                        (52, 69, 147), (100, 115, 255), (0, 24, 236),
                        (132, 56, 255), (82, 0, 133), (203, 56, 255),
                        (255, 149, 200), (255, 55, 199)],
                       dtype=np.float32) / 255.
ALPHA = 0.5