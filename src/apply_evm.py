import cv2
from evm.pyramid import get_gaussian_pyramid
from evm.filter import bandpass_filter
from heart_rate import find_heart_rate
from constants import BUFFER_SIZE, FREQ_MIN, FREQ_MAX, FPS, LEVELS
import numpy as np
import threading
from collections import deque
import time

def apply_evm(buffer: deque, stopEvent: threading.Event):

    freqs = np.fft.fftfreq(BUFFER_SIZE, d=1.0 / FPS)
    mask = (freqs >= FREQ_MIN) & (freqs <= FREQ_MAX)

    while(True):

        if(stopEvent.is_set()):
            break


        if(len(buffer) == BUFFER_SIZE):

            frames = np.asarray(buffer).astype(np.float32)

            start_pyramid = time.time()
            pyramid = []
            for frame in frames:
                frame_green_channel = frame[:, :, 1]
                pyramid.append(get_gaussian_pyramid(frame_green_channel, LEVELS))
            end_pyramid = time.time()


            start_filter = time.time()
            filtered_fft = bandpass_filter(pyramid, mask)
            end_filter = time.time() 

            heart_rate = find_heart_rate(filtered_fft, freqs)

            print(f"Pyramid {(end_pyramid - start_pyramid)*1000:.2f} ms | Filter {(end_filter - start_filter)*1000:.2f} ms  | Heart rate: {heart_rate}")


    
            
        