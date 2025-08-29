#from scipy.signal import find_peaks
import numpy as np
import cupy as cp

def find_heart_rate(fft, freqs):
    magnitude = cp.abs(fft)
    mean_magnitude = magnitude.mean(axis=(1, 2))
    peak_freq = freqs[cp.argmax(mean_magnitude).item()]
    heart_rate = peak_freq * 60.0
    return heart_rate
