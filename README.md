# yolorun

testing about yolo inference


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


## tensorrt working in progress


FUNZIONA: create onnx from *.pt 
```
task ultra
yolo export model=/models/yolov8n.pt format=onnx simplify=True [opset=12]


task trt
#cd /app/yolorun/experiments/TensorRT-For-YOLO-Series
export-det -o /models/yolov8n.onnx -e /models/yolov8n.engine --end2end --v8
#python trt.py -e /models/yolov8n.engine --end2end -i /samples/0a0a00b2fbe89a47.jpg -o /samples/out.jpg
yolorun --model /models/yolov8n.engine --show --step /samples/*jpg


# non funziona
trtexec --onnx=/models/yolov8n.onnx --saveEngine=/models/yolov8n.engine --fp16
```


DA VERIFICARE
```
task ultra
yolo export model=/models/yolov8n.pt format=onnx simplify=True 
```

create engine from onnx (run on same GPU that has to make inference)
```
task trt
trtexec --onnx=/models/yolov8n.onnx --saveEngine=/models/yolov8n.engine --fp16
```

inference
```
task trt
yolorun --model /models/yolov8n.engine --show --step /samples/*jpg
```

