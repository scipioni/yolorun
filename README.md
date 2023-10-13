# yolorun

testing about yolo inference: ```yolorun -m <model> --size 416```

| 416x416 | darknet yolov3 supertiny with cv.dnn  | yolov8n with ultralitycs | yolov8n with nms in trt/pycuda |  |
|---|---|---|---|---|
| quadro 4000 | 260fps inference 3.2ms | 114fps | 191fps inference 2.7ms |  |
| RTX 3060 | 290fps inference 3.3ms |  |  |  |

## GPU

prereq
```
yay -S libnvidia-container libnvidia-container-tools nvidia-container-toolkit
sudo systemctl restart docker
```

check capabilities with
```
nvidia-smi --query-gpu=compute_cap --format=csv
```


## build

```
task build
```

## cli

```
task cli
```

## convert yolo segmentation dataset into label studio dataset


Create a labelstudio source storage with path: /dataset/plates-01 

Create project.js to import in labelstudio
```
yolo2ls --out /tmp/project.js --prefix plates-01 /dataset/*txt
```

## create onnx model

from pt format
```
yolo export model=yolov8n.pt imgsz=640 format=onnx opset=12
```

## autolabel of dataset

```
yolorun --model models/yolov8x.pt /tmp/openimages/*jpg --filter-classes 0,56,57,59,60 --save /archive/dataset/
```

## filter a dataset

any
```
yolorun --save /tmp/dataset-with-2-or-3 --filter-classes 2,3 /dataset-orig/*jpg

```


strict
```
yolorun --save /tmp/dataset-with-only-2-3 --filter-classes-strict 2,3 /dataset-orig/*jpg

```

## convert classes.txt

```
find ./train -name "*txt" | xargs -n1 sed -i 's/^0/80/; s/^1/81/; s/^2/57/; s/^3/59/'
```

## example

```
# autolabel di persone, letti, sofa, tavoli e sedie
yolorun --model models/yolov8x.pt /archive/dataset/fp/train/sofabed/*jpg  --filter-classes 0,56,57,59,60 --save /archive/dataset/fp/sofabed.autolabel
```

rtsp on thermal camera
```
IP=10.1.8.171
yolorun -m /models/fp.engine --show rtsp://$IP:554/cam/realmonitor?channel=2&subtype=0&unicast=true&proto=Onvif
```

## tensorrt 


### detection


create onnx from *.pt 
```
task ultra
yolo export model=/models/yolov8n.pt format=onnx simplify=True imgsz=416 [opset=12] [half=True] [dynamic=True]
```

create *.engine from *.onnx
```
task trt
export-det -o /models/yolov8n.onnx --end2end --v8
```

inference with trt/pycuda
```
yolorun --model /models/yolov8n.engine --show --step /samples/*jpg


# non funziona
# trtexec --onnx=/models/yolov8n.onnx --saveEngine=/models/yolov8n.engine --fp16
```


### segmentation

create onnx from *.pt
```
yolo export model=/models/yolov8n-seg.pt format=onnx  imgsz=416 simplify=True [opset=12] [simplify=True]
```

create *.engine from *.onnx
```
task trt
export-seg -o /models/yolov8n-seg.onnx 
```
inference with trt/pycuda
```
task trt
yolorun --model /models/yolov8n-seg.engine --show --step /samples/*jpg
```


## Taotoolkit Segmentation Model Training and Deployment

1) TaoToolkit:
- https://docs.nvidia.com/tao/tao-toolkit/text/tao_toolkit_quick_start_guide.html#running-tao-toolkit
- [Sample Workflows](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/resources/cv_samples)

2) Dataset:
- https://docs.nvidia.com/tao/tao-toolkit/text/data_annotation_format.html#semantic-segmentation-format
- https://docs.nvidia.com/tao/tao-toolkit/text/semantic_segmentation/segformer.html#dataset-config-segformer

3) Training:
- https://docs.nvidia.com/tao/tao-toolkit/text/semantic_segmentation/segformer.html

4) Deploy:
- [To TensorRT](https://docs.nvidia.com/tao/tao-toolkit/text/tao_deploy/segformer.html)
- [To Deepstream](https://docs.nvidia.com/tao/tao-toolkit/text/ds_tao/segformer_ds.html#deploying-to-deepstream-segformer)