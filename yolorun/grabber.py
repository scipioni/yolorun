import configargparse
import cv2 as cv
import numpy as np
from . import settings


def initParser():
    parser = configargparse.get_argument_parser()
    parser.add_argument("images", nargs="+", help="list of images")
    parser.add_argument("--show", action="store_true", default=False)
    parser.add_argument("--step", action="store_true", default=False)
    parser.add_argument("--model", default="/models/plates-seg.onnx")
    return parser.parse_args()


def main():
    config = settings.get()
    for filename in config.images:
        frame = cv.imread(filename)

        if config.show:
            cv.imshow("image", frame)
            key = cv.waitKey(0 if config.step else 1)
            if key in (ord("q"), 27):
                break


if __name__ == "__main__":
    main()
