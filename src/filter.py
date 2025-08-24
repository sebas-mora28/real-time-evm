import numpy as np
from constants import FREQ_MIN, FREQ_MAX

def bandpass_filter(frames_buffer, mask):
    fft = np.fft.fft(np.asarray(frames_buffer), axis=0)
    fft[~mask, :, :] = 0
    return fft
