import cv2
import numpy as np
import os

def load_yolo_model(config_path, weights_path, classes_path):
    net = cv2.dnn.readNet(weights_path, config_path)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    
    with open(classes_path, "r") as f:
        classes = [line.strip() for line in f.readlines()]

    return net, output_layers, classes

def detect_objects(img, net, output_layers):
    height, width, channels = img.shape
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    boxes = []
    confidences = []
    class_ids = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    return boxes, confidences, class_ids

def draw_labels(boxes, confidences, class_ids, classes, img):
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = (0, 255, 0)
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y - 10), font, 2, color, 2)

def process_frames(frame_folder, output_folder, net, output_layers, classes):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for frame_file in os.listdir(frame_folder):
        img_path = os.path.join(frame_folder, frame_file)
        img = cv2.imread(img_path)
        boxes, confidences, class_ids = detect_objects(img, net, output_layers)
        draw_labels(boxes, confidences, class_ids, classes, img)
        
        cv2.imwrite(os.path.join(output_folder, frame_file), img)

if __name__ == "__main__":
    # Required YOLO files
    config_path = "models/yolo/yolov3.cfg"
    weights_path = "models/yolo/yolov3.weights"
    classes_path = "models/yolo/coco.names"
    
    # Build YOLO model
    net, output_layers, classes = load_yolo_model(config_path, weights_path, classes_path)
    
    # Path to input frames
    frame_folder = "data/processed_frames/"
    # Path to output folder
    output_folder = "data/detection_results/"
    
    # Run detection on all frames
    process_frames(frame_folder, output_folder, net, output_layers, classes)
    
    print(f"Detection completed. Processed frames saved in {output_folder}")
