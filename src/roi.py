import cv2
import numpy as numpy

face_cascade = cv2.CascadeClassifier("/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml")

def detect_roi(frame):
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
  frame_roi = None
  if len(faces) > 0:
    (x, y, w, h) = faces[0] 
    frame_roi = frame[y: y+h, x: x+w]
    frame_roi = cv2.resize(frame_roi, (320, 320))
    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
  return frame_roi