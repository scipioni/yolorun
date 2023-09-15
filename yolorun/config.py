import random

import numpy as np

from typing import Any
import configargparse
import logging


def get_config() -> Any:
    parser = configargparse.get_argument_parser()

    #parser.add_argument("--step", action="store_true", default=False, help="step mode")


    parser.add_argument("--debug", action="store_true", default=False, help="debugging mode")
    parser.add_argument(
        "--config", required=False, is_config_file=True, help="config file path"
    )
    parser.add_argument(
        "--config-save",
        required=False,
        is_write_out_config_file_arg=True,
        help="config file path",
    )

    parser.add_argument("images", nargs="*", default=[], help="list of images")
    parser.add_argument("--show", action="store_true", default=False)
    parser.add_argument("--dummy", action="store_true", default=False)
    parser.add_argument("--step", action="store_true", default=False)
    parser.add_argument("--crop", action="store_true", default=False)
    
    #parser.add_argument("--models", default="./models", help="models repo")
    parser.add_argument("--model", default="", help="model name")
    parser.add_argument("--show-ann", action="store_true", default=False, help="show annotator info")
    parser.add_argument("--url", default="", help="camera url, for example rtsp://10.1.16.107:554/s0")
    parser.add_argument("--confidence-min", type=float, default=0.1)

    #parser.add_argument('--engine', default="models/yolov8n.engine", help='Engine file')
    parser.add_argument('--device',
                        default='cuda:0',
                        help='TensorRT infer device')

    parser.add_argument("--filter-classes", default="", help="match given classes, for example 0,1,...")
    parser.add_argument("--filter-classes-strict", default="", help="match given classes, for example 0,1,...")
    # parser.add_argument("--inference", action="store_true", default=False)
    # parser.add_argument("--nano", action="store_true", default=True)
    # parser.add_argument("--custom", action="store_true", default=False)


    # parser.add_argument("--quiet", action="store_true", default=False)
    # parser.add_argument(
    #     "--delay-start",
    #     type=int,
    #     default=0,
    #     help="add a delay at start (needed for truth brain tests)",
    # )
    # parser.add_argument("--camera-framerate", type=int, default=25)
    # parser.add_argument("--camera-cols", type=int, default=2048)
    # parser.add_argument("--camera-rows", type=int, default=1024)
    # parser.add_argument("--camera-roi-cols", type=int, default=1024)
    # parser.add_argument("--camera-roi-rows", type=int, default=512)
    # parser.add_argument("--camera-top", type=int, default=256)
    # parser.add_argument("--camera-shutter", type=float, default=1.0, help="shutter time in ms")
    # parser.add_argument(
    #     "--camera-AutoGainUpperLimit", type=float, default=23.0, help="AutoGainUpperLimit"
    # )
    # parser.add_argument(
    #     "--camera-AutoTargetBrightness",
    #     type=float,
    #     default=0.25,
    #     help="AutoTargetBrightness",
    # )
    # parser.add_argument(
    #     "--camera-AutoExposureTimeUpperLimit",
    #     type=float,
    #     default=5000,
    #     help="AutoExposureTimeUpperLimit",
    # )
    # parser.add_argument("--camera-color", action="store_true", default=False, help="")
    # parser.add_argument("--camera-native", action="store_true", default=False, help="")
    # parser.add_ar gument("--camera-basler", action="store_true", default=True, help="")
    # parser.add_argument("--camera-loop", action="store_true", default=False)
    # parser.add_argument("--camera-fake", default="", help="simulate camera with image")
    parser.add_argument("--camera-id", nargs="+", default=[0])
    # parser.add_argument("--camera-serial", default="")  # default="23255275")
    # parser.add_argument("--camera-flip-y", type=int, default=0, help="flip vertical")
    # parser.add_argument("--socket-port", type=int, default=5007)
    # parser.add_argument("--socket-ip", default="127.0.0.1")
    # parser.add_argument("--socket-ip-gps", default="127.0.0.1")
    # parser.add_argument("--socket-port-gps", type=int, default=5012)
    # parser.add_argument(
    #     "--gps-sync",
    #     action="store_true",
    #     default=False,
    #     help="emit GPS sync on UDP port --socket-port-gps",
    # )
    # # parser.add_argument('--socket-multicast-group', default='224.1.1.1')
    # parser.add_argument("--path", default=os.getcwd())
    # parser.add_argument("--save", action="store_true", default=False)
    # parser.add_argument("--save-path", default="/tmp")
    # parser.add_argument(
    #     "--save-encoder",
    #     default="x264enc speed-preset=7",
    #     help="h264 encoder [x264enc, omxh264enc control-rate=2 bitrate=10000000]",
    # )
    # parser.add_argument("--legacy", action="store_true", default=False, help="test legacy code")
    # parser.add_argument("--check-shm", type=int, default=30, help="check shm allocation every n frames; 0 disable check")
    parser.add_argument("--move", default="", help="move files to <path>")
    parser.add_argument("--save", default="", help="autolabeling to <path>")
    parser.add_argument("--merge", help="merge prediction with ground truthclea", action="store_true", default=False)


    config = parser.parse_args()
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s %(message)s",
        level="DEBUG" if config.debug else "INFO",
    )

    if config.filter_classes:
        config.filter_classes = [int(cls) for cls in config.filter_classes.split(",")]
    if config.filter_classes_strict:
        config.filter_classes_strict = [int(cls) for cls in config.filter_classes_strict.split(",")]

    return config


random.seed(0)

# detection model classes
CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
           'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
           'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
           'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
           'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
           'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
           'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
           'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
           'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
           'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
           'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
           'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
           'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
           'scissors', 'teddy bear', 'hair drier', 'toothbrush')

# colors for per classes
COLORS = {
    cls: [random.randint(0, 255) for _ in range(3)]
    for i, cls in enumerate(CLASSES)
}

# colors for segment masks
MASK_COLORS = np.array([(255, 56, 56), (255, 157, 151), (255, 112, 31),
                        (255, 178, 29), (207, 210, 49), (72, 249, 10),
                        (146, 204, 23), (61, 219, 134), (26, 147, 52),
                        (0, 212, 187), (44, 153, 168), (0, 194, 255),
                        (52, 69, 147), (100, 115, 255), (0, 24, 236),
                        (132, 56, 255), (82, 0, 133), (203, 56, 255),
                        (255, 149, 200), (255, 55, 199)],
                       dtype=np.float32) / 255.

# alpha for segment masks
ALPHA = 0.5
