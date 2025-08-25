from collections import deque
import cv2
from constants import BUFFER_SIZE, FREQ_MIN, FREQ_MAX, LEVELS, FPS
from helpers import show_frame
from filter import bandpass_filter
from heartrate import find_heart_rate
from pyramid import get_gaussian_pyramid
from roi import detect_roi
import time
import matplotlib.pyplot as plt
import numpy as np

gst_str = ("v4l2src device=/dev/video0 ! "
           "video/x-raw, width=640, height=480, framerate=30/1 ! "
           "videoconvert ! video/x-raw, format=BGR ! appsink")


# Crear objeto de captura de video con GStreamer
cam = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)

frames_buffer = deque(maxlen=BUFFER_SIZE)

freqs = np.fft.fftfreq(BUFFER_SIZE, d=1.0 / FPS)
mask = (freqs >= FREQ_MIN) & (freqs <= FREQ_MAX)

heart_rate_buffer = deque(maxlen=30)

while(True):
	
    ret, frame = cam.read()
    
    if not ret:
      break

    start = time.time()

    frame_roi = detect_roi(frame)
    
    if(frame_roi is not None):
      frame_green_channel = frame[:, :, 1] 
      pyramid = get_gaussian_pyramid(frame_green_channel, LEVELS)
      frames_buffer.append(pyramid)

    if(len(frames_buffer) == BUFFER_SIZE):

      print("El buffer està lleno")

      filtered_fft = bandpass_filter(frames_buffer, mask) 
	
      heart_rate = find_heart_rate(filtered_fft, freqs)
      heart_rate_buffer.append(heart_rate)

    show_frame(frame, np.asarray(heart_rate_buffer).mean())


    print("Time: ", time.time() - start)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cam.release()
cv2.destroyAllWindows()





