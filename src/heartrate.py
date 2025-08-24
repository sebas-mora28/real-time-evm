#from scipy.signal import find_peaks
import numpy as np

def find_heart_rate(fft, freqs):
    magnitude = np.abs(fft)
    mean_magnitude = magnitude.mean(axis=(1, 2))
    peak_freq = freqs[np.argmax(mean_magnitude)]
    heart_rate = peak_freq * 60.0
    return heart_rate
