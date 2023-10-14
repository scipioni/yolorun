# YOLOv8-ONNX-RKNN-HORIZON-TensorRT-Segmentation
***Remark: This repo only support 1 batch size***
![!YOLOv8 ONNX RKNN Segmentation Picture](https://github.com/laitathei/YOLOv8-ONNX-RKNN-Segmentation/blob/master/doc/visual_image.jpg)
![!YOLOv8 ONNX RKNN Segmentation Video](https://github.com/laitathei/YOLOv8-ONNX-RKNN-Segmentation/blob/master/doc/result.gif)

Video source: https://www.youtube.com/watch?v=n3Dru5y3ROc&t=0s
```
git clone --recursive https://github.com/laitathei/YOLOv8-ONNX-RKNN-HORIZON-TensorRT-Segmentation.git
```
## 0. Environment Setting
```
# For onnx, rknn, horizon
torch: 1.10.1+cu102
torchvision: 0.11.2+cu102
onnx: 1.10.0
onnxruntime: 1.10.0

# For tensorrt
torch: 1.11.0+cu113
torchvision: 0.12.0+cu113
TensorRT: 8.6.1
```

## 1. Yolov8 Prerequisite
```
pip3 install ultralytics==8.0.147
pip3 install numpy==1.23.5
```

## 2. Convert Pytorch model to ONNX
Remember to change the variable to your setting.
```
python3 pytorch2onnx.py
```


## 9. Onnx Runtime Inference
```
python3 onnxruntime_inference.py
```

## 10. Convert ONNX model to TensorRT 
Remember to change the variable to your setting
```
python3 onnx2trt.py
```

## 11. TensorRT Inference
```
python3 tensorrt_inference.py
```

## 12. Blob Inference
Convert model from onnx to blob format via ```https://blobconverter.luxonis.com/```
```
python3 blob_inference.py
```

## Reference
```
https://blog.csdn.net/magic_ll/article/details/131944207
https://blog.csdn.net/weixin_45377629/article/details/124582404#t18
https://github.com/ibaiGorordo/ONNX-YOLOv8-Instance-Segmentation
```
