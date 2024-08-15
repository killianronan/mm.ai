**MMA Analysis Project**

This project aims to build a computer vision-based system to analyze Mixed Martial Arts (MMA) matches using deep learning techniques. The system includes components for fighter detection, strike recognition, and metric collection on fights, all based on video input.

- Special thanks to the MMA community for inspiring this project.

**Project Overview**
The goal of this project is to create an automated system that can:

- Detect and track fighters in MMA videos using YOLO.
- Estimate the poses of the fighters to analyze movements.
- Classify different types of strikes, takedowns, and other actions using TensorFlow.
- Collect metrics on fight performance, such as control time and strike effectiveness.

**Features**
- Object Detection: Detect and track fighters in video frames.
- Pose Estimation: Analyze keypoints on fighters' bodies to understand their movements.
- Action Recognition: Classify actions (e.g., punches, kicks, takedowns) from video sequences.
- Metric Collection: Collect and analyze metrics related to fight performance.

**Technologies**
- YOLO for object detection.
- TensorFlow for deep learning.
- OpenCV for video processing.

**Steps to Run Pipeline**
Download the YOLO weights file from: https://github.com/patrick013/Object-Detection---Yolov3/blob/master/model/yolov3.weights.
Add the downloaded file to the models/yolo folder, this file is too large to push to Github (250MB, limit is 100MB).

Install dependencies:
pip install -r requirements.txt

Extract frames from video:
python src/data_processing/extract_frames.py

Perform image detection using YOLO on the frames:
python src/detection/yolo_detector.py

View the results under detection_results folder

**Training Models**
jupyter notebook notebooks/model_training.ipynb

**Data**

**License**

This project is licensed under the MIT License - see the LICENSE file for details.
