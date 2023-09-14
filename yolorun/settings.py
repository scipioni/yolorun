import configargparse

def get():
    parser = configargparse.get_argument_parser()
    parser.add_argument("images", nargs="+", help="list of images")
    parser.add_argument("--show", action="store_true", default=False)
    parser.add_argument("--step", action="store_true", default=False)
    parser.add_argument("--model", default="/models/plates-seg.onnx")
    return parser.parse_args()

