#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
from yolodetect import detect_largest_aeroplane

FRAMES = 65
COLOR = (0, 0, 255)
FONT_SCALE = 3.0
THICKNESS = 5
K = 0.7
H = int(1080*K)
W = int(1920*K)
RESIZE_DIMS = (W, H)
MODE_TEXT_POS = (3840-1650,2160-200, )


def create_tracker(tracker_type):
    """
    Create an OpenCV tracker based on the given type.
    """
    if tracker_type == 'MIL':
        return cv2.legacy.TrackerMIL_create()
    elif tracker_type == 'KCF':
        return cv2.legacy.TrackerKCF_create()
    elif tracker_type == "CSRT":
        return cv2.legacy.TrackerCSRT_create()
    else:
        raise ValueError(f"Unknown tracker type: {tracker_type}")

def initialize_tracker(cap, tracker, x1, y1, width, height):
    """
    Initialize the tracker with the first frame from the video capture.
    """
    ret, img = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        exit()

    bbox = (x1, y1, width, height)
    tracker.init(img, bbox)

    return img, bbox

def draw_rectangle(img, bbox, color=(0, 255, 0), thickness=2):
    """
    Draw a rectangle around the tracked object.
    """
    x1, y1 = int(bbox[0]), int(bbox[1])
    width, height = int(bbox[2]), int(bbox[3])
    cv2.rectangle(img, (x1, y1), (x1 + width, y1 + height), color, thickness)

def process_frame_with_tracker(tracker, img, tracker_type):
    """
    Process the frame using the tracker and update the bounding box.
    """
    ok, bbox = tracker.update(img)
    if ok:
        draw_rectangle(img, bbox)
        x, y, w, h = [int(v) for v in bbox]
        cv2.putText(img, tracker_type, (x + w, y + h), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, COLOR, THICKNESS)
    return ok

def process_frame_with_detector(img):
    """
    Process the frame using the YOLO detector to find the largest aeroplane.
    """
    bbox = detect_largest_aeroplane(img)
    if bbox:
        draw_rectangle(img, bbox)            
        x, y, w, h = bbox
        cv2.putText(img, "AERO", (x + w, y + h), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, COLOR, THICKNESS)
    return bbox

def display_frame(img, resize_dimensions=RESIZE_DIMS):
    """
    Display the frame with resized dimensions.
    """
    stretch_near = cv2.resize(img, resize_dimensions, interpolation=cv2.INTER_LINEAR)
    cv2.imshow('frame', stretch_near)

def detecting(video_path):
    """
    detecting function to run the detector on the video.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Cannot open video")
        exit()

    tracker_frame_counter = FRAMES
    while tracker_frame_counter > 0:
        ret, img = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        bbox = process_frame_with_detector(img)
        if not bbox:
            cv2.putText(img, "Detecting FAIL", (100, 150), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, COLOR, THICKNESS)
        else:
            cv2.putText(img, "Detecting SUCCESS", (100, 150), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, COLOR, THICKNESS)
        cv2.putText(img, "Working only DETECTING", MODE_TEXT_POS, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, COLOR, THICKNESS)
        display_frame(img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        tracker_frame_counter -= 1

    cap.release()
    cv2.destroyAllWindows()

def tracking(video_path, tracker_type, x1, y1, x2, y2):
    """
    tracking function to run the tracker on the video.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Cannot open video")
        exit()
    
    width, height = x2 - x1, y2 - y1
    tracker = create_tracker(tracker_type)
    img, bbox = initialize_tracker(cap, tracker, x1, y1, width, height)

    tracker_frame_counter = FRAMES
    while tracker_frame_counter > 0:
        ret, img = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        ok = process_frame_with_tracker(tracker, img, tracker_type)
        if not ok:
            cv2.putText(img, "Tracking FAIL", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, COLOR, THICKNESS)
        else:
            cv2.putText(img, "Tracking SUCCESS", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, COLOR, THICKNESS)
        cv2.putText(img, "Working only TRACKING", MODE_TEXT_POS, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, COLOR, THICKNESS)
        display_frame(img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        tracker_frame_counter -= 1

    cap.release()
    cv2.destroyAllWindows()

def tracking_detecting(video_path, tracker_type, x1, y1, x2, y2):
    """
    tracking_detecting function to run the tracker on the video and help it with yolo detecting.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Cannot open video")
        exit()
    
    width, height = x2 - x1, y2 - y1
    tracker = create_tracker(tracker_type)
    img, bbox = initialize_tracker(cap, tracker, x1, y1, width, height)

    tracker_frame_counter = FRAMES
    while tracker_frame_counter > 0:
        ret, img = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        ok = process_frame_with_tracker(tracker, img, tracker_type)
        if ok:
            cv2.putText(img, "Tracking SUCCESS", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, COLOR, THICKNESS)
        else:
            cv2.putText(img, "Tracking FAIL", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, COLOR, THICKNESS)
            bbox = process_frame_with_detector(img)
            if not bbox:
                cv2.putText(img, "Detecting FAIL", (100, 180), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, COLOR, THICKNESS)
            else:
                tracker = create_tracker(tracker_type)
                tracker.init(img, bbox)
                cv2.putText(img, "Detecting SUCCESS", (100, 180), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, COLOR, THICKNESS)
        cv2.putText(img, "Working only TRACKING+DETECTING", MODE_TEXT_POS, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, COLOR, THICKNESS)
        display_frame(img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        tracker_frame_counter -= 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_paths = ["dron1.mov", "dron2.mp4", "dron3.mov", "dron4.mov"]
    tracker_types = ['KCF', 'CSRT', 'MIL']
    rects = [
        (1377, 1499, 1479, 1557),
        (675, 1445, 918, 1572),
##        (685, 1450, 908, 1562),
        (1347, 1817, 1443, 1876),
##        (1357, 1827, 1433, 1866),
        (2439, 1455, 2579, 1510)
    ]
    start_from = 0

    # USE TRACKING AND IF IT NEEDS TRY DETECT
    for video_path, rect in zip(video_paths[start_from:], rects[start_from:]):
        x1, y1, x2, y2 = rect
        for tracker_type in tracker_types:
            tracking_detecting(video_path, tracker_type, x1, y1, x2, y2)  

    
    # USE ONLY TRACKING
    for video_path, rect in zip(video_paths[start_from:], rects[start_from:]):
        x1, y1, x2, y2 = rect
        for tracker_type in tracker_types:
            tracking(video_path, tracker_type, x1, y1, x2, y2)




    #JUST YOLO AEROPLANE DETECTOR
    for video_path in video_paths[start_from:]:
        detecting(video_path) 

    


