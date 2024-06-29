import cv2
import numpy as np
import time

# Paths to the YOLO files
config_path = 'yolo/yolov3.cfg'
weights_path = 'yolo/yolov3.weights'
names_path = 'yolo/coco.names'

# Load class names
classes = open(names_path).read().strip().split('\n')

# Check if the class "aeroplane" is in the list of classes
if "aeroplane" not in classes:
    print("Error: 'aeroplane' class not found in the provided names file.")
    exit()

aeroplane_class_id = classes.index("aeroplane")

# Load YOLO network
net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)

# Get the output layer names
ln = net.getUnconnectedOutLayersNames()

def detect_largest_aeroplane(img):
    if img is None:
        print("Error: Could not open or find the image.")
        return None

    h, w = img.shape[:2]

    # Prepare the image for detection
    blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    # Perform detection
    t0 = time.time()
    outputs = net.forward(ln)
    t = time.time()
    print(f'It took {t - t0:.3f} seconds to process the image.')

    # Process detections
    boxes = []
    confidences = []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if class_id == aeroplane_class_id and confidence > 0.5:
                box = detection[:4] * np.array([w, h, w, h])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))

    # Apply non-maxima suppression to filter out weak detections
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Find the largest bounding box
    if len(indices) > 0:
        largest_box = None
        largest_area = 0
        for i in indices.flatten():
            (x, y, w, h) = boxes[i]
            area = w * h
            if area > largest_area:
                largest_area = area
                largest_box = (x, y, w, h)

        if largest_box:
            (x, y, w, h) = largest_box
            # Crop the image to the largest bounding box
##            cropped_img = img[y:y + h, x:x + w]
            return largest_box

    return None

if __name__ == "__main__":
    img_path = 'drone.png'
    # Load the image
    img = cv2.imread(img_path)
    cropped_img = detect_largest_aeroplane(img)