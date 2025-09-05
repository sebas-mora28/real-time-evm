import numpy as np


def bandpass_filter(images, fps, freq_min, freq_max, axis=0):
    fft = np.fft.fft(images, axis=axis)
    frequencies = np.fft.fftfreq(images.shape[0], d=1.0/fps)

    print(frequencies)

    low = (np.abs(frequencies - freq_min)).argmin()
    high = (np.abs(frequencies - freq_max)).argmin()

    fft[:low] = 0
    fft[high:] = 0

    return fft, frequencies

