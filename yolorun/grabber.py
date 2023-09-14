import configargparse
import cv2 as cv
import numpy as np
from . import settings
from .lib.timing import timing, Counter
from .lib.yoloseg_gorordo import YOLOSeg

def initParser():
    parser = configargparse.get_argument_parser()
    parser.add_argument("images", nargs="+", help="list of images")
    parser.add_argument("--show", action="store_true", default=False)
    parser.add_argument("--step", action="store_true", default=False)
    parser.add_argument("--model", default="/models/plates-seg.onnx")
    parser.add_argument("--backend", default="onnxruntime")
    return parser.parse_args()






def main():
    config = settings.get()

    with timing("load model"):
        yoloseg = YOLOSeg(config.model, conf_thres=0.3, iou_thres=0.5)

    with timing("total"):
        counter = Counter()
        for filename in config.images:
            frame = cv.imread(filename)

            with counter("inference"):
                yoloseg(frame)

            frame_out = yoloseg.draw_masks(frame)

            if config.show:
                cv.imshow("image", frame_out)
                key = cv.waitKey(0 if config.step else 1)
                if key in (ord("q"), 27):
                    break
        print(counter)

if __name__ == "__main__":
    main()
