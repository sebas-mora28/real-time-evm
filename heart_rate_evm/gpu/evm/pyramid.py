import cv2
from cupyx.scipy.ndimage import gaussian_filter


def get_gaussian_pyramid_gpu(frames, levels):
    current = frames
    for i in range(levels):
        blurred = gaussian_filter(current, sigma=(0, 1, 1))  # adjust sigma
        current = blurred[:, ::2, ::2]  # downsample by factor of 2
    return current