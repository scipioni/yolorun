# Build TensorRT engine from Onnx file
import os
import sys
import logging
import argparse
import os, time, json
from datetime import datetime
import tensorrt as trt


def main(args):
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    config.max_workspace_size = args.workspace * 1 << 30 # in terms of GB

    # #config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace * (2 ** 30)) # in terms of GB
    flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flag)
    parser = trt.OnnxParser(network, logger)

    if not os.path.exists(args.onnx) or not parser.parse_from_file(args.onnx):
        raise RuntimeError(f'failed to load ONNX file: {args.onnx}')

    inputs = [network.get_input(i) for i in range(network.num_inputs)]
    outputs = [network.get_output(i) for i in range(network.num_outputs)]

    print("Network Description")
    for input in inputs:
        print("Input '{}' with shape {} and dtype {}".format(input.name, input.shape, input.dtype))
    for output in outputs:
        print("Output '{}' with shape {} and dtype {}".format(output.name, output.shape, output.dtype))

    if builder.platform_has_fast_fp16 and args.precision == "fp16": # build with fp16
        config.set_flag(trt.BuilderFlag.FP16)
    elif builder.platform_has_fast_int8 and args.precision == "int8": # build with int8
        config.set_flag(trt.BuilderFlag.INT8)

    start = time.time()
    # metadata = {
    #     'description': "custom model (untrained)",
    #     'date': datetime.now().isoformat(),
    #     'version': "0.1",
    #     'stride': 32,
    #     'task': 'segment',
    #     'batch': 1,
    #     'imgsz': [1, 1],
    #     'names': {"0":"plate"}}  # model metadata

    # # Write file
    with builder.build_serialized_network(network, config) as engine, open(args.engine, 'wb') as f:
        # Metadata
        #meta = json.dumps(metadata)
        #f.write(len(meta).to_bytes(4, byteorder='little', signed=True))
        #f.write(meta.encode())
        # Model
        f.write(engine) #.serialize())
    elapsed = time.time()-start
    print(f"saved {args.engine} in {elapsed} seconds")

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--onnx", help="The input ONNX model file to load")
    parser.add_argument("-e", "--engine", default="", help="The output path for the TRT engine")
    parser.add_argument("-p", "--precision", default="fp16", choices=["fp16", "int8"],
                        help="The precision mode to build in, either 'fp16' or 'int8', default: 'fp16'")
    # parser.add_argument("-v", "--verbose", action="store_true", help="Enable more verbose log output")
    parser.add_argument("-w", "--workspace", default=1, type=int, help="The max memory workspace size to allow in Gb, "
                                                                        "default: 1")
    # parser.add_argument("--calib_input", help="The directory holding images to use for calibration")
    # parser.add_argument("--calib_cache", default="./calibration.cache",
    #                     help="The file path for INT8 calibration cache to use, default: ./calibration.cache")
    # parser.add_argument("--calib_num_images", default=5000, type=int,
    #                     help="The maximum number of images to use for calibration, default: 5000")
    # parser.add_argument("--calib_batch_size", default=8, type=int,
    #                     help="The batch size for the calibration process, default: 8")
    # parser.add_argument("--end2end", default=True, action="store_true",
    #                     help="export the engine include nms plugin, default: True")
    # parser.add_argument("--conf_thres", default=0.4, type=float,
    #                     help="The conf threshold for the nms, default: 0.4")
    # parser.add_argument("--iou_thres", default=0.5, type=float,
    #                     help="The iou threshold for the nms, default: 0.5")
    # parser.add_argument("--max_det", default=100, type=int,
    #                     help="The total num for results, default: 100")
    # parser.add_argument("--v8", default=True, action="store_true",
    #                     help="use yolov8 model, default: True")
    args = parser.parse_args()
    print(args)
    if not args.engine:
        args.engine = args.onnx.replace(".onnx", ".engine")
    main(args)

if __name__ == "__main__":
    run()