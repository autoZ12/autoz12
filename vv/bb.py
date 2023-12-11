import cv2

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face = face_detector.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in face:
        cv2.rectangle(frame, (x,y), (x+y,y+h), (255,0,0), 2)



    cv2.imshow('frame', frame)
    cv2.waitKey(1)
