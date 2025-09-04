import cv2
from collections import deque
import threading
from helpers import show_frame

gst_str = ("v4l2src device=/dev/video0 ! "
           "video/x-raw, width=640, height=480, framerate=30/1 ! "
           "videoconvert ! video/x-raw, format=BGR ! appsink")


# Crear objeto de captura de video con GStreamer
cam = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)

def capture_frames(buffer: deque, result, stop_event):

  while(True):
	
    ret, frame = cam.read()
    if not ret:
      break

    buffer.append(frame)
    show_frame(frame, result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        stop_event.set()
        break

  cam.release()
  cv2.destroyAllWindows()
