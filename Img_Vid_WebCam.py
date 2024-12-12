import cv2
import argparse
import numpy as np


# Define classes and COLORS globally
classes = None
COLORS = None

ap = argparse.ArgumentParser()
ap.add_argument('-v', '--video', help='/opencv_project/object-detection-opencv/video.mp4')
ap.add_argument('-i', '--image', help = '/opencv_project/object-detection-opencv/image.png')
ap.add_argument('-wb', '--webcam', action='store_true', help='use webcam for live object detection')
ap.add_argument('-c', '--config', required=True, help = '/opencv_project/object-detection-opencv/yolov3.cfg')
ap.add_argument('-w', '--weights', required=True, help = '/opencv_project/yolov3.weights')
ap.add_argument('-cl', '--classes', required=True, help = '/opencv_project/object-detection-opencv/yolov3.txt')
args = ap.parse_args()

def find_output_layers(model):
    layer_names = model.getLayerNames()
    out_layers = model.getUnconnectedOutLayers()
    # Handling both flat and nested array cases:
    if out_layers.ndim == 1:
        output_layers = [layer_names[i - 1] for i in out_layers]
    else:
        output_layers = [layer_names[i[0] - 1] for i in out_layers.flatten()]
    return output_layers

def draw_prediction(img, class_id, confidence, box):
    if classes is not None:  # Check if classes is initialized
        label = str(classes[class_id])
        color = COLORS[class_id]
        cv2.rectangle(img, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), color, 2)
        cv2.putText(img, f"{label}: {confidence:.2f}", (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    else:
        print("Classes are not initialized. Unable to draw prediction.")

def process_image(image_path):
    image = cv2.imread(image_path)
    height, width = image.shape[:2]

    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(find_output_layers(net))

    process_detections(image, height, width, outs)

    cv2.imshow("Object Detection", image)
    cv2.waitKey(0)
    cv2.imwrite("object-detection-output.jpg", image)
    cv2.destroyAllWindows()

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        height, width = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(find_output_layers(net))

        process_detections(frame, height, width, outs)

        cv2.imshow("Object Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def process_webcam():
    net = cv2.dnn.readNet(args.weights, args.config)
    cap = cv2.VideoCapture(0)  # 0 for default webcam
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        height, width = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(find_output_layers(net))
        process_detections(frame, height, width, outs)
        cv2.imshow("Object Detection - Webcam", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def process_detections(frame, height, width, outs):
    conf_threshold = 0.5
    nms_threshold = 0.4
    boxes, confidences, class_ids = [], [], []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                center_x, center_y = int(detection[0] * width), int(detection[1] * height)
                w, h = int(detection[2] * width), int(detection[3] * height)
                x, y = center_x - w // 2, center_y - h // 2
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    if len(indices) > 0:  # Check if indices is not empty
        for i in indices.flatten():
            draw_prediction(frame, class_ids[i], confidences[i], boxes[i])
    else:
        print("No objects detected.")

with open(args.classes, 'r') as file:
    classes = [line.strip() for line in file.readlines()]

COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

#net = cv2.dnn.readNet(args.weights, args.config)


if args.image:
    net = cv2.dnn.readNet(args.weights, args.config)
    process_image(args.image)
elif args.video:
    net = cv2.dnn.readNet(args.weights, args.config)
    process_video(args.video)
elif args.webcam:
    process_webcam()
else:
    print("Please specify an image, video path, or use --webcam for webcam input.")
