Real-Time Object Detection with Faster R-CNN and YOLOv8

This project explores two of the most influential object detection paradigms in Deep Learning:

Faster R-CNN (Two-Stage Detector)
YOLOv8n (One-Stage Detector)

The objective is to compare detection accuracy, computational efficiency, and real-time performance using pretrained models evaluated on a filtered subset of the MS COCO dataset containing the classes:

Person
Chair
Laptop

In addition to the experimental evaluation, a real-time object detection system was developed to perform inference directly from a webcam, allowing dynamic switching between both architectures.

Project Objectives
Compare one-stage and two-stage object detectors.
Analyze the trade-off between detection accuracy and inference speed.
Evaluate pretrained models using COCO metrics (mAP).
Deploy a real-time object detection application.
Models Evaluated
Faster R-CNN ResNet50-FPN-V2

A two-stage detector that first generates object proposals using a Region Proposal Network (RPN) and then refines detections through classification and bounding-box regression.

Characteristics:

43.7 million parameters
High localization accuracy
Suitable for offline analysis and inspection tasks
YOLOv8n

A lightweight one-stage detector that performs object localization and classification in a single forward pass.

Characteristics:

3.2 million parameters
Real-time performance
Optimized for CPU deployment
Dataset

The experiments were conducted using a filtered subset of the Microsoft COCO dataset.

Selected classes:

Person
Chair
Laptop

Instead of fine-tuning, an inference-only strategy was adopted because both models were already pretrained on the complete COCO dataset, which contains tens of thousands of annotations for these classes.

Technologies
Python
PyTorch
Torchvision
Ultralytics YOLOv8
OpenCV
pycocotools
NumPy
Matplotlib
Experimental Results
Model	mAP@0.5	mAP@0.5:0.95	FPS (CPU)	Parameters
Faster R-CNN	0.698	0.467	3.2	43.7M
YOLOv8n	0.524	0.372	45.0	3.2M
Key Findings
Faster R-CNN achieved the highest detection accuracy.
YOLOv8n was approximately 14× faster on CPU.
YOLOv8n is the most practical solution for real-time applications.
Faster R-CNN is preferable when detection precision is the primary objective.
Real-Time Detection System

The project includes a webcam-based application with:

Real-time object detection
Dynamic model switching
Confidence threshold adjustment
Snapshot capture
Video recording mode
FPS monitoring
Example Usage
python deteccion_tiempo_real.py --modelo yolo

python deteccion_tiempo_real.py --modelo faster --conf 0.5

python deteccion_tiempo_real.py --grabar --segundos 30
Conclusions

The experiments demonstrate the classical trade-off between accuracy and speed in object detection systems. Faster R-CNN provides superior localization performance, while YOLOv8n delivers real-time inference with significantly lower computational requirements. For practical deployment on CPU devices, YOLOv8n represents the best balance between performance and efficiency.
