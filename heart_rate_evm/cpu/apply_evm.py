from evm.pyramid import get_gaussian_pyramid
from evm.filter import bandpass_filter
from core.heart_rate import find_heart_rate
from core.roi import detect_roi
from core.constants import BUFFER_SIZE, FREQ_MIN, FREQ_MAX, FPS, LEVELS
import numpy as np
import threading
from collections import deque
import time

buffer_bpm = deque(maxlen=30)


def apply_evm(buffer: deque, result: deque, stopEvent: threading.Event):

    freqs = np.fft.fftfreq(BUFFER_SIZE, d=1.0 / FPS)
    mask = (freqs >= FREQ_MIN) & (freqs <= FREQ_MAX)


    while(True):

        if(stopEvent.is_set()):
            print("Finshing thread execution")
            break

        if(len(buffer) == BUFFER_SIZE):

            start_time = time.time()

            frames = list(buffer)
            first_frame = frames[0]
            roi = detect_roi(first_frame)

            if(roi is None):
                print("No face detected")
                continue

            (x, y, w, h) = roi
            roi_frames = [frame[y:y+h, x:x+w] for frame in frames]
            roi_frames = np.asarray(roi_frames).astype(np.float32)
           
            start_pyramid = time.time()
            green_channel_frames = roi_frames[:, :, :, 1]
            pyramid = get_gaussian_pyramid(green_channel_frames, LEVELS)

            # pyramid = []
            # for frame in roi_frames:
            #     frame_green_channel = frame[:, :, 1]
            #     pyramid.append(get_gaussian_pyramid(frame_green_channel, LEVELS))
            end_pyramid = time.time()

            start_filter = time.time()
            filtered_fft = bandpass_filter(pyramid, mask)
            end_filter = time.time() 


            start_heart_rate = time.time()
            heart_rate = find_heart_rate(filtered_fft, freqs)
            end_heart_rate = time.time()

            end_time = time.time()

            time_elapsed = (end_time - start_time)*1000
            print(f"Heart rate: {heart_rate} | Time elapsed: {time_elapsed:.2f} | Pyramid {(end_pyramid - start_pyramid)*1000:.2f} ms | Filter {(end_filter - start_filter)*1000:.2f} ms | Heart rate Calculation {(end_heart_rate - start_heart_rate)*1000:.2f} ms")


            buffer_bpm.append(heart_rate)

            avg_bpm = int(np.asarray(list(buffer_bpm)).mean())

            result.append({"heart_rate": avg_bpm, "roi": roi})

            

            print(f"")


    
            
        