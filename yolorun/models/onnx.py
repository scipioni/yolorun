# https://github.com/ultralytics/ultralytics/blob/main/examples/YOLOv8-OpenCV-ONNX-Python/main.py

import cv2 as cv
import numpy as np

from .__init__ import Model
from yolorun.lib.grabber import BBoxes, BBox
from yolorun.lib.timing import timing
import onnxruntime as ort







class ModelOnnxRuntime(Model):
    def __init__(self, config):
        super().__init__(config)
        self.session = ort.InferenceSession(config.model, providers=['TensorrtExecutionProvider'])
        self.model_inputs = self.session.get_inputs()

        # Store the shape of the input for later use
        input_shape = self.model_inputs[0].shape
        self.channels = input_shape[1]
        self.size = (input_shape[2], input_shape[3])

    def __str__(self):
        return "  %s size=%sx%sx%d CUDA=%s" % (
            self.config.model,
            self.size[0],
            self.size[1],
            self.channels,
            not self.config.cpu,
        )

    def preprocess(self, img):
        # Convert the image color space from BGR to RGB
        img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)

        # Resize the image to match the input shape
        img = cv.resize(img, self.size)

        # Normalize the image data by dividing it by 255.0
        image_data = np.array(img) / 255.0

        # Transpose the image to have the channel dimension as the first dimension
        image_data = np.transpose(image_data, (2, 0, 1))  # Channel first

        #######
        #img = img.astype(np.float32)
        #######

        # Expand the dimensions of the image data to match the expected input shape
        image_data = np.expand_dims(image_data, axis=0).astype(np.float32)

        # Return the preprocessed image data
        return image_data

    def postprocess(self, output):
        """
        Performs post-processing on the model's output to extract bounding boxes, scores, and class IDs.

        Args:
            input_image (numpy.ndarray): The input image.
            output (numpy.ndarray): The output of the model.

        Returns:
            numpy.ndarray: The input image with detections drawn on it.
        """

        # Transpose and squeeze the output to match the expected shape
        outputs = np.transpose(np.squeeze(output[0]))

        # Get the number of rows in the outputs array
        rows = outputs.shape[0]

        # Lists to store the bounding boxes, scores, and class IDs of the detections
        boxes = []
        scores = []
        class_ids = []

        # Calculate the scaling factors for the bounding box coordinates
        x_factor = self.w / self.size[0]
        y_factor = self.h / self.size[1]

        # Iterate over each row in the outputs array
        for i in range(rows):
            # Extract the class scores from the current row
            classes_scores = outputs[i][4:]

            # Find the maximum score among the class scores
            max_score = np.amax(classes_scores)

            # If the maximum score is above the confidence threshold
            if max_score >= self.confidence_thres:
                # Get the class ID with the highest score
                class_id = np.argmax(classes_scores)

                # Extract the bounding box coordinates from the current row
                x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]

                # Calculate the scaled coordinates of the bounding box
                left = int((x - w / 2) * x_factor)
                top = int((y - h / 2) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)

                # Add the class ID, score, and box coordinates to the respective lists
                class_ids.append(class_id)
                scores.append(max_score)
                boxes.append([left, top, width, height])

        # Apply non-maximum suppression to filter out overlapping bounding boxes
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.confidence_thres, self.iou_thres)

        # Iterate over the selected indices after non-maximum suppression
        for i in indices:
            # Get the box, score, and class ID corresponding to the index
            box = boxes[i]
            score = scores[i]
            class_id = class_ids[i]

            # Draw the detection on the input image
            #self.draw_detections(input_image, box, score, class_id)
        return indices, boxes, scores, class_ids


    def predict(self, frame):
        super().predict(frame)
        img_data = self.preprocess(frame)
        outputs = session.run(None, {self.model_inputs[0].name: img_data})
        indices, boxes, confidences, classIds = self.postprocess(frame, outputs)  # output image


        for i in indices:
            # Get the box, score, and class ID corresponding to the index

            left, top, width, height = boxes[i]
            if confidences[i] < self.config.confidence_min:
                continue

            self.bboxes.add(
                BBox(
                    classIds[i],
                    left,
                    top,
                    left + width,
                    top + height,
                    self.w,
                    self.h,
                    confidences[i],
                )
            )