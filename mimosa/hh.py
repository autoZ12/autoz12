import cv2
import parinya
cap = cv2.VideoCapture(0)
yolo = parinya.YOLOv3('coco.names.txt, yolov3-tiny.cfg.txt, yolov3-tiny.weights')
while True:
    _, frame = cap.read()
    yolo.detect(frame, draw=False)
    for d in object:
        label, left, top, width, height = d
        print(d)
    cv2.imshow('frame', frame)
    cv2.waitKey(1)
