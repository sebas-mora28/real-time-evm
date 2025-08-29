
import cv2
import numpy as np
import time
from constants import  BUFFER_SIZE
from helpers import load_video
from evm.filter import bandpass_filter
from evm.laplacian_pyramid import build_laplacian_pyramids, build_laplacian_pyramid_gpu
from evm.gaussian_pyramid import build_gaussian_pyramids
from heart_rate import find_heart_rate
import matplotlib.pyplot as plt

# Initialize video capture
# cap = cv2.VideoCapture(0)

# frame_buffer = []
# timestamps = []
# roi_signal = []

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Convert to YCrCb or RGB
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     frame_resized = cv2.resize(frame_rgb, (160, 120))

#     # Extract ROI (face region - simplified as center region)
#     h, w, _ = frame_resized.shape
#     roi = frame_resized[h//4:h//2, w//4:w//2, :]

#     # Compute mean green channel (most sensitive to pulse)
#     green_channel = np.mean(roi[:, :, 1])
#     roi_signal.append(green_channel)

#     timestamps.append(time.time())
#     if len(roi_signal) > BUFFER_SIZE:
#         roi_signal.pop(0)
#         timestamps.pop(0)

#         # Bandpass filter for heart rate frequencies
#         b, a = signal.butter(1, [LOW_CUT/(FPS/2), HIGH_CUT/(FPS/2)], btype='band')
#         filtered_signal = signal.filtfilt(b, a, roi_signal)

#         # Find peaks to estimate BPM
#         peaks, _ = signal.find_peaks(filtered_signal, distance=FPS/2)
#         if len(peaks) > 1:
#             intervals = np.diff([timestamps[p] for p in peaks])
#             bpm = 60.0 / np.mean(intervals)
#         else:
#             bpm = 0

#         # Display heart rate on screen
#         cv2.putText(frame, f"HR: {int(bpm)} BPM", (20, 50),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

#     cv2.imshow("Real-Time EVM HR", frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()



path = '../data/sebastian-cerca-3.mp4'
#path = '../data/face.png'
levels = 4
freq_min = 1
freq_max = 1.8
alpha = 100

#frames, fps = load_video(path)
#print("Images shape: ", frames.shape)


#pyramids = build_laplacian_pyramids(images, levels)

face_cascade = cv2.CascadeClassifier("/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


while True:
  ret, frame = cap.read()
  if not ret:
    break

  print("Size: ", face_cascade.empty()) 

  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)


  if len(faces) > 0:
    for (x, y, w, h) in faces:
      cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

  print("Pasa")
  cv2.imshow("face detection", frame)

  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

  #start_time_gpu = time.time()
  #gpu_frame = cv2.cuda_GpuMat()
  #gpu_frame.upload(frame.astype(np.float32))
  #frame_size = (gpu_frame.size()[1], gpu_frame.size()[0])
  #result = cv2.cuda.dft(gpu_frame, frame_size)

  #print(result.size())
  
  #for _ in range(levels):
  #  gpu_frame = cv2.cuda.pyrDown(gpu_frame)
  #gpu_frame.download()
  #end_time_gpu = time.time()

  #gpu_time = end_time_gpu - start_time_gpu

  #start_time_cpu = time.time()
  #result = np.fft.fft(frame)
  #print(result.shape)
  #for _ in range(levels):
  #  frame = cv2.pyrDown(frame)
  #end_time_cpu = time.time()
  #cpu_time = end_time_cpu - start_time_cpu
 
  #winner = "GPU" if cpu_time > gpu_time else "CPU"
    

  #y = cpu_time if cpu_time > gpu_time else gpu_time
  #x = (abs(gpu_time - cpu_time) / y) * 100

  #print("GPU Time: ", gpu_time, " CPU Time: ", cpu_time, " Winner: ", winner, " % ", x)
  
#start_time_pyr = time.time()
#pyramids  = build_laplacian_pyramids_gpu(frames, levels)
#end_time_pyr = time.time()
#print("Pyramid time: ", end_time_pyr - start_time_pyr)
#print("Size pyramid: ", len(pyramids), " Level shape: ", pyramids[0].shape)



#start_time_filter = time.time()
#fft, freqs = bandpass_filter(pyramids[-2], freq_min, freq_max)
#end_time_filter = time.time()
#print("Filter: ", end_time_filter - start_time_filter)
#print("FFT shape: ", fft.shape, "    ", "Frecuencias shape: ", freqs.shape)

#heart_rate_bpm = find_heart_rate(fft, freqs, 1, 1.8)

#print(f"Heart Rate: {heart_rate_bpm} BPM")


# heart_rate = find_heart_rate(fft, freqs, freq_min, freq_max)
# print("Hear rate: ", heart_rate)


# import cv2
# #import mediapipe as mp
# import numpy as np

# #mp_face = mp.solutions.face_detection
# #detector = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.7)


# cap = cv2.VideoCapture(0)
# fps = 30
# buffer_size = 150
# temporal_buffer = []
# alpha = 50
# levels = 4
# low_freq, high_freq = 0.4, 3.0

# def bandpass_filter(signal, low, high, fps):
#     freqs = np.fft.rfftfreq(len(signal), d=1/fps)
#     fft_signal = np.fft.rfft(signal, axis=0)
#     fft_signal[(freqs < low) | (freqs > high)] = 0
#     return np.fft.irfft(fft_signal, axis=0)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     #results = detector.process(frame_rgb)

#     if results.detections:
#         h, w, _ = frame.shape
#         box = results.detections[0].location_data.relative_bounding_box
#         x1, y1 = int(box.xmin * w), int(box.ymin * h)
#         x2, y2 = x1 + int(box.width * w), y1 + int(box.height * h)

#         roi = frame[y1:y2, x1:x2]
#         if roi.size > 0:
#             roi_small = cv2.resize(roi, (64, 64))
#             temporal_buffer.append(cv2.cvtColor(roi_small, cv2.COLOR_BGR2GRAY))
#             if len(temporal_buffer) > buffer_size:
#                 temporal_buffer.pop(0)

#             if len(temporal_buffer) == buffer_size:
#                 filtered = bandpass_filter(np.array(temporal_buffer), low_freq, high_freq, fps)
#                 amplified = filtered[-1] * alpha
#                 amplified = cv2.resize(amplified, (roi.shape[1], roi.shape[0]))
#                 amplified = np.expand_dims(amplified, axis=-1)
#                 amplified = np.repeat(amplified, 3, axis=-1)
#                 roi = np.clip(roi + amplified, 0, 255).astype(np.uint8)

#             frame[y1:y2, x1:x2] = roi

#     cv2.imshow("EVM con Detección de Cara", frame)
#     if cv2.waitKey(1) & 0xFF == 27:
#         break

cap.release()
cv2.destroyAllWindows()










