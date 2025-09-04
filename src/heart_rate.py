#from scipy.signal import find_peaks
import numpy as np
import cupy as cp
from constants import FREQ_MIN, FREQ_MAX
from scipy.signal import find_peaks

# def find_heart_rate(fft, freqs):
#     magnitude = cp.abs(fft)
#     mean_magnitude = magnitude.mean(axis=(1, 2))
#     peak_freq = freqs[cp.argmax(mean_magnitude).item()]
#     heart_rate = peak_freq * 60.0
#     return heart_rate


def find_heart_rate(fft, freqs):
    fft = fft.get()
    fft_maximums = []
    for i in range(len(freqs)):
        if FREQ_MIN <= freqs[i] <= FREQ_MAX:
            fft_value = abs(fft[i])
            fft_maximums.append(fft_value.max())
        else:
            fft_maximums.append(0)

    peaks, _ = find_peaks(fft_maximums)
    max_peak = -1
    max_frequency = 0
    for peak in peaks:
        if fft_maximums[peak] > max_frequency:
            max_frequency = fft_maximums[peak]
            max_peak = peak

    return freqs[max_peak] * 60




