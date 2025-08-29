import numpy as np
from constants import FREQ_MIN, FREQ_MAX
import cupy as cp

def bandpass_filter(frames, mask):
    frames_gpu = cp.asarray(frames)
    fft = cp.fft.fft(frames_gpu, axis=0)
    cp.cuda.Stream.null.synchronize() 
    fft[~mask, :, :] = 0
    return fft
