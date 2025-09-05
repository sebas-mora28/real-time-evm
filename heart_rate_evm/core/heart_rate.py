from .constants import FREQ_MIN, FREQ_MAX
from scipy.signal import find_peaks

def find_heart_rate(fft, freqs):
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




