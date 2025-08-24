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


cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


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





