import cv2
from collections import deque
from core.helpers import show_frame
import numpy as np
from core.roi import detect_roi
from core.heart_rate import find_heart_rate
from .evm.pyramid import generate_gaussian_pyramids
from .evm.filter import bandpass_filter
from core.constants import FREQ_MIN, FREQ_MAX, FPS, LEVELS


cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

def capture_frames(buffer: deque, result, stop_event):

  buffer = deque(maxlen=240)
  result = deque(maxlen=1)

  while(True):
	
    ret, frame = cam.read()
    if not ret:
      break

    buffer.append(frame)


    if(len(buffer) == 240):
      frames = list(buffer)
      # first_frame = frames[0]
      # roi = detect_roi(first_frame)
      # if(roi is None):
      #     print("No face detected")
      #     continue
      # (x, y, w, h) = roi
      #roi_frames = [frame[y:y+h, x:x+w] for frame in frames]
      #roi_frames = np.asarray(frames).astype(np.float32)
      
      #print(np.asarray(frames).shape)
      #green_channel_frames = frames[:, :, :, 1]

      pyramids = generate_gaussian_pyramids(frames, LEVELS)
    
      filtered_fft, freqs = bandpass_filter(pyramids, FPS, FREQ_MIN, FREQ_MAX)
      heart_rate = find_heart_rate(filtered_fft, freqs)

      buffer.clear()

      print("Heart rate: ", heart_rate)
      result = {"heart_rate": heart_rate}
    
    show_frame(frame, result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        stop_event.set()
        break

  cam.release()
  cv2.destroyAllWindows()
